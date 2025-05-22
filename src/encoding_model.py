"""Predict the EEG responses using features from DNN models

Parameters
----------
sub : int
    Subject identifier.
tot_eeg_chan : int
    Total number of EEG channels to model.
tot_eeg_time : int
    Total number of EEG time points per trial.
vision_model : str
    Name of the vision model to use (e.g., 'cornet_s').
language_model : str
    Name of the language model to use (e.g., 'text_embedding_large').
nfeature_vm : int
    Number of principal components to retain from vision features.
nfeature_lm : int
    Number of principal components to retain from language features.
project_dir : str
    Root directory for all input data and output results.
tag : str
    Version tag for differentiating result output files.
metrics : str
    Regression metric to optimize ('r2', 'mse', or 'mae').
vision_only : bool
    If True, run image-only encoding.
language_only : bool
    If True, run text-only encoding.
fusion : bool
    If True, run joint image-text fusion encoding.
partition : bool
    If True, run the script on separate CPUs in parallel
n_split : int
    Number of parallel splits for time slicing.
n : int
    Zero-based index of the current split to process.

"""

import argparse
import os
import os.path as op
import sys
import numpy as np
import random
from utils import Grid_search,Grid_search_fusion


parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, default=4, help='Subject identifier')
parser.add_argument('--tot_eeg_chan', type=int, default=63, help='Total number of EEG channels')
parser.add_argument('--tot_eeg_time', type=int, default=180, help='Total number of EEG time points per trial')
parser.add_argument('--vision_model', type=str, default='cornet_s', help='Vision model name')
parser.add_argument('--language_model', type=str, default='text_embedding_large', help='Language model name')
parser.add_argument('--nfeature_vm', type=int, default=1000, help='Number of vision feature components')
parser.add_argument('--nfeature_lm', type=int, default=1000, help='Number of language feature components')
parser.add_argument('--project_dir', type=str, default='/scratch/byrong/encoding/data', help='Root project directory')
parser.add_argument('--tag', type=str, default='v3', help='Version tag for output files')
parser.add_argument('--metrics', type=str, choices=['r2','mse','mae'], default='r2', help='Regression metric to optimize')
parser.add_argument('--vision_only', action='store_true', help='Run visio-only encoding')
parser.add_argument('--language_only', action='store_true', help='Run text-only encoding')
parser.add_argument('--fusion', action='store_true', help='Run image-text fusion encoding')
parser.add_argument('--partition', action='store_true', help='Run the script on separate CPUs')
parser.add_argument('--n_split', type=int, default=90, help='Number of parallel time splits')
parser.add_argument('--n', type=int, default=0, help='Zero-based split index to process')
args = parser.parse_args()

print('>>> Ridge encoding <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


random.seed(42)
np.random.seed(42)

embedding_file = 'gpt4_features_embedded_by_%s_pca.npy'%(args.language_model)
print(embedding_file)


### Loading the embeded text features ###
data_dict = np.load(op.join(args.project_dir,'gpt4_features', embedding_file), allow_pickle=True).item()
language_embedding_train = data_dict['embedding_train'].astype(np.float32)
train_index = data_dict['train_index']
language_embedding_test = data_dict['embedding_test'].astype(np.float32)

# Average across five versions if not 
if language_embedding_train.shape[0] == 5:
    language_embedding_train = np.average(language_embedding_train,axis = 0)
    language_embedding_test = np.average(language_embedding_test,axis = 0)
# Take the first n principle component for language embeddings            
n = language_embedding_train.shape[-1]
if n > args.nfeature_lm:
    language_embedding_train = language_embedding_train[:,:args.nfeature_lm]
    language_embedding_test = language_embedding_test[:,:args.nfeature_lm]

channels = 'preprocessed_eeg_data_v1'


### Loading the EEG training data ###
data_dir = os.path.join('eeg_dataset',channels)
training_file = 'eeg_'+ 'sub-'+ format(args.sub,'02') +'_split-train.npy'

data = np.load(os.path.join(args.project_dir, data_dir, training_file),
        allow_pickle=True).item()
y_train = data['preprocessed_eeg_data'].astype(np.float32)
ch_names = data['ch_names']
times = data['times']
y_train = y_train[train_index]

# Averaging across repetitions
y_train = np.mean(y_train, 1)[:,:args.tot_eeg_chan]
### Loading the EEG test data ###

test_file = 'eeg_'+ 'sub-'+ format(args.sub,'02') +'_split-test.npy'
data = np.load(os.path.join(args.project_dir, data_dir, test_file),
  allow_pickle=True).item()
y_test = data['preprocessed_eeg_data'].astype(np.float32)
# Averaging across repetitions
y_test = np.mean(y_test, 1)
y_test = y_test[:,:args.tot_eeg_chan]


feature_path = op.join(args.project_dir,'pca_feature_maps/%s/pretrained-True/layers-all/pca_feature_maps_training.npy'%args.vision_model)
vision_feature_train = np.load(feature_path,allow_pickle=True).item()['all_layers'].astype(np.float32)
feature_path = op.join(args.project_dir,'pca_feature_maps/%s/pretrained-True/layers-all/pca_feature_maps_test.npy'%args.vision_model)
vision_feature_test = np.load(feature_path,allow_pickle=True).item()['all_layers'].astype(np.float32)

vision_feature_train = vision_feature_train[:,:args.nfeature_vm]
vision_feature_test = vision_feature_test[:,:args.nfeature_vm]
vision_feature_train = vision_feature_train[train_index]

# identify alpha ranges to search for
alpha_range = np.logspace(-3,7,100)

if args.partition:
    ## select the slices of data for distributing computation on different cpus
    args.tot_eeg_time = args.tot_eeg_time//args.n_split
    y_train = y_train[:,:,args.n*args.tot_eeg_time:(args.n+1)*args.tot_eeg_time]
    y_test = y_test[:,:,args.n*args.tot_eeg_time:(args.n+1)*args.tot_eeg_time]
    times = times[args.n*args.tot_eeg_time:(args.n+1)*args.tot_eeg_time]
else:
    args.n = 'all'

# Initialize predictions and results        
y_pred_vision_only = np.zeros(y_test.shape)
y_pred_fusion = np.zeros(y_test.shape)
y_pred_language_only = np.zeros(y_test.shape)
alpha_vision_only,alpha_fusion,alpha_language_only = [], [], []
weight_fusion = []
flag_fusion = []

print("after slicing:",args.tot_eeg_time, y_train.shape,y_test.shape)

save_dir = os.path.join(args.project_dir, 'linear_results', 'sub-'+
                  format(args.sub,'02'), 'synthetic_eeg_data')

if not op.exists(save_dir):
    os.makedirs(save_dir)

# iterative grid search loops for vision_only, language_only, and fusion
for t in range(args.tot_eeg_time-1):
    for c in range(args.tot_eeg_chan):
        
        if args.vision_only:

            y_pred_vision_only[:,c,t], alpha, flag = Grid_search(vision_feature_train,vision_feature_test,y_train[:,c,t],y_test[:,c,t],
                                        alpha_range = alpha_range,
                                        metrics = args.metrics,kfold = 5)

            alpha_vision_only.append(alpha)

            if (c == args.tot_eeg_chan-1) and (t == args.tot_eeg_time-1):
                data_dict = {
                    'synthetic_data' : y_pred_vision_only,
                    'ch_names' : ch_names,
                    'times': times,
                  'alpha': alpha_vision_only
                }

                
                file_name = '%s_%s_%s_%s.npy'%(args.vision_model,args.metrics,args.tag,args.n)

                np.save(os.path.join(save_dir, file_name), data_dict)

            
    


        if args.language_only:

            y_pred_language_only[:,c,t], alpha, flag = Grid_search(language_embedding_train,language_embedding_test,y_train[:,c,t],y_test[:,c,t],
                                              alpha_range = alpha_range,
                                              metrics = args.metrics,kfold = 5)
            alpha_language_only.append(alpha) 

            if (c == args.tot_eeg_chan-1) and (t == args.tot_eeg_time-1):
                data_dict = {
                'synthetic_data' : y_pred_language_only,
                'ch_names' : ch_names,
                'times': times,
                'alpha': alpha_language_only,
                 }

                
                    
                file_name = '%s_%s_%s_%s.npy'%(args.language_model,args.metrics,args.tag,args.n)

                np.save(os.path.join(save_dir, file_name), data_dict)


       


        # fusing representations

        if args.fusion:
           
            
            y_pred_fusion[:,c,t], alpha, weight,flag = Grid_search_fusion(vision_feature_train,vision_feature_test,language_embedding_train,language_embedding_test,y_train[:,c,t],y_test[:,c,t],
                        alpha_range = alpha_range,weight_list = np.linspace(0,1000,41)/1000,
                        metrics = args.metrics,kfold = 5)
        

            alpha_fusion.append(alpha)
            weight_fusion.append(weight)
            flag_fusion.append(flag)
            if (c == args.tot_eeg_chan-1) and (t == args.tot_eeg_time-1):
                data_dict = {
                'synthetic_data' : y_pred_fusion,
                'ch_names' : ch_names,
                'times': times,
                'alpha': alpha_fusion,
                'weight': weight_fusion,
                'flag':flag_fusion
                 }

                    
                file_name = '%s_with_%s_%s_%s.npy'%(args.vision_model,args.language_model,args.metrics,args.tag,args.n)

                np.save(os.path.join(save_dir, file_name), data_dict)
