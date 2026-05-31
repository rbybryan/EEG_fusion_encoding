"""Training end-to-end DNN models which predict the EEG responses to the test
images. The EEG responses to each test image condition are synthesized in a
cross-validated fashion, by using the training epoch weights which yielded the
highest correlation prediction score for all the other N-1 test image
conditions.

Parameters
----------
sub : int
	Used subject.
tot_eeg_chan : int
	Total amount of EEG channels.
tot_eeg_time : int
	Total amount of EEG time points.
dnn : str
	DNN model used.
project_dir : str
	Directory of the project folder.

"""

import numpy as np
import os
import os.path as op
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
import tqdm
from sklearn.utils import resample
from scipy.stats import pearsonr,spearmanr
from matplotlib import pyplot as plt
import random
import time
from sklearn.metrics.pairwise import cosine_similarity
import argparse

# --- path configuration (portable defaults for public release) ---
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_DATA_ROOT = _os.environ.get('EEG_FUSION_DATA', _os.path.join(_REPO, 'data'))
# --- end path configuration ---


parser = argparse.ArgumentParser()
parser.add_argument('--text_embedding_file', type=str, default='gpt4_features_embedded_large.npy')
parser.add_argument('--project_dir', default=_DATA_ROOT, type=str)
args = parser.parse_args()


### Loading the embeded text features ###
data_dict = np.load(op.join(args.project_dir, 'gpt4_features', args.text_embedding_file), allow_pickle=True).item()


train_index = data_dict["train_index"].copy()
text_train = data_dict['text_features_train_long'].copy()
del data_dict



def random_sample(all_sample_size,sample_size,seed = 2024):
    random.seed(seed)
    # Assuming you have a list of conditions/images
    all_conditions = np.arange(all_sample_size)  # assuming 10,000 conditions


    # Shuffle the list of conditions/images
    random.shuffle(all_conditions)

    # Number of images to sample each time
    sample_size = 100

    # Initialize an empty list to store the sampled images
    samples = []

    # Iterate over the shuffled list in chunks of sample_size
    for i in range(0, len(all_conditions), sample_size):
        # Get a chunk of sample_size images
        chunk = all_conditions[i:i + sample_size]
        # Add the chunk to the list of sampled images
        if len(chunk) == sample_size:
            samples.append(chunk)

    # Print or use sampled images
    #print(sampled_images)
    return samples

channels = 'preprocessed_data_all' 
sample_size = 100
row_indices, col_indices = np.tril_indices(sample_size,-1)
t_win = 5
subject_corr = {}
for sub in range(1,11): 
    if sub != 5:
       
        tt = time.time()
        # load eeg data
        data_dir = os.path.join('eeg_dataset', channels, 'sub-'+
            format(sub,'02'))
        training_file = 'preprocessed_eeg_training.npy'
        data = np.load(os.path.join(args.project_dir, data_dir, training_file),
            allow_pickle=True).item()
        ch_names = data['ch_names']
        times = data['times']
        bio_train = data['preprocessed_eeg_data']
        bio_train = bio_train[train_index]
        # Averaging across repetitions
        y_train = np.mean(bio_train, 1)
        # Converting to float32 (for DNN training with Pytorch)
        y_train = np.float32(y_train)[:,:63,:]
        # y_test = y_test.reshape(-1,17*100
        samples = random_sample(len(y_train),sample_size,sub)
        
        corr = {}
        corr['spearman_cos_eeg_cos_emb'] = np.zeros((len(samples),63,100-t_win))
        corr['spearman_cor_eeg_cos_emb'] = np.zeros((len(samples),63,100-t_win))
        # corr['spearman_cos_eeg_cos_img'] = np.zeros((len(samples),63,90))
        # corr['spearman_cor_eeg_cor_img'] = np.zeros((len(samples),63,90))
        corr['pearson_cos_eeg_cos_emb'] = np.zeros((len(samples),63,100-t_win))
        corr['pearson_cor_eeg_cos_emb'] = np.zeros((len(samples),63,100-t_win))
        # corr['pearson_cos_eeg_cos_img'] = np.zeros((len(samples),63,90))
        # corr['pearson_cor_eeg_cor_img'] = np.zeros((len(samples),63,90))

#         cos_emb,cos_eeg,cor_eeg,cos_img,cor_img = [],[],[],[],[]

        for n,s in enumerate(samples): 
            sampled_eeg = y_train[s]
            sampled_emb = text_train[s]
            # sampled_img = img_train[s]
            cos_emb = cosine_similarity(sampled_emb)[row_indices, col_indices]
#             cos_img = cosine_similarity(sampled_img)[row_indices, col_indices]
#             cor_img = np.corrcoef(sampled_img)[row_indices, col_indices]
# #             cos_eeg_tmp,cor_eeg_tmp = [], []
            for c in range(63):
                cos_eeg_tmp_ch,cor_eeg_tmp_ch = [], []
                for t, tp in enumerate(range(0,100-t_win)):
                    cos_eeg = cosine_similarity(sampled_eeg[:,c,tp:tp+t_win])[row_indices, col_indices]
                    cor_eeg = np.corrcoef(sampled_eeg[:,c,t:t+t_win])[row_indices, col_indices]
                    corr['spearman_cos_eeg_cos_emb'][n,c,t] = spearmanr(1-cos_eeg,1-cos_emb)[0]
                    corr['spearman_cor_eeg_cos_emb'][n,c,t] = spearmanr(1-cor_eeg,1-cos_emb)[0]
                    # corr['spearman_cos_eeg_cos_img'][n,c,t] = spearmanr(1-cos_eeg,1-cos_img)[0]
                    # corr['spearman_cor_eeg_cor_img'][n,c,t] = spearmanr(1-cor_eeg,1-cor_img)[0]
                    corr['pearson_cos_eeg_cos_emb'][n,c,t] = pearsonr(1-cos_eeg,1-cos_emb)[0]
                    corr['pearson_cor_eeg_cos_emb'][n,c,t] = pearsonr(1-cor_eeg,1-cos_emb)[0]
                    # corr['pearson_cos_eeg_cos_img'][n,c,t] = pearsonr(1-cos_eeg,1-cos_img)[0]
                    # corr['pearson_cor_eeg_cor_img'][n,c,t] = pearsonr(1-cor_eeg,1-cor_img)[0]  
              
        print('subject ',sub, ' time elapsed: ',time.time()-tt, 's')
        subject_corr[sub] = corr        

save_dir = os.path.join(args.project_dir, 'RSA')

if not op.exists(save_dir):
    os.makedirs(save_dir)
    
file_name = 'Channel_wise_%s_per_%d.npy'%(args.text_embedding_file,t_win)

np.save(os.path.join(save_dir, file_name), subject_corr)









