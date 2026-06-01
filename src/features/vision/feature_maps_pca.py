"""Apply PCA on the DNN feature maps to reduce their dimensionality.

PCA is applied on either the feature maps of single DNN layers, or on the
appended feature maps of all layers. Before applying PCA on the CORnet-S
feature maps, run ``sort_feature_maps_cornet_s.py``.

Parameters
----------
dnn : str
    Used DNN among 'alexnet', 'resnet50', 'cornet_s', 'moco'.
pretrained : bool
    If True use the pretrained network feature maps, if False use the randomly
    initialized network feature maps.
layers : str
    Whether to use 'all' or 'single' layers.
n_components : int
    Number of DNN feature maps PCA components retained.
project_dir : str
    Directory of the project folder.
"""

import argparse
import os
import os.path as op
import pickle

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--dnn', default='resnet50_clip', type=str,
                    help='Used DNN feature maps to apply PCA on.')
parser.add_argument('--pretrained', default=True, type=str,
                    help='Whether to use pretrained network feature maps.')
parser.add_argument('--layers', default='all', type=str,
                    help="Whether to use 'all' or 'single' layers.")
parser.add_argument('--n_components', default=1000, type=int,
                    help='Number of PCA components retained.')
parser.add_argument('--project_dir', default='/scratch/byrong/encoding/data',
                    type=str, help='Directory of the project folder.')
args = parser.parse_args()

print('>>> Apply PCA on the feature maps <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220

print(args.pretrained)

if args.pretrained == 'False':
    args.pretrained = False
else:
    args.pretrained = True
print(args.pretrained)


# =============================================================================
# Apply PCA on the training images feature maps
# =============================================================================
# The standardization and PCA statistics computed on the training images feature
# maps are also applied to the test images feature maps and to the ILSVRC-2012
# images feature maps.

# Load the feature maps
feats = []
fmaps_train = {}
fmaps_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
                         'full_feature_maps', args.dnn,
                         'pretrained-' + str(args.pretrained),
                         'training_images')
print(fmaps_dir)
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
for f, fmaps in enumerate(fmaps_list):
    fmaps_data = np.load(os.path.join(fmaps_dir, fmaps),
                         allow_pickle=True).item()
    all_layers = fmaps_data.keys()
    if args.layers == 'all':
        layer_names = ['all_layers']
    elif args.layers == 'single':
        layer_names = all_layers
    for l, dnn_layer in enumerate(all_layers):
        if args.layers == 'all':
            if l == 0:
                feats = np.reshape(fmaps_data[dnn_layer], -1)
            else:
                feats = np.append(feats, np.reshape(fmaps_data[dnn_layer], -1))
        elif args.layers == 'single':
            if f == 0:
                feats.append([[np.reshape(fmaps_data[dnn_layer], -1)]])
            else:
                feats[l].append([np.reshape(fmaps_data[dnn_layer], -1)])
    if args.layers == 'all':
        if f == 0:
            print(feats.shape)
            feats_all = np.empty((len(fmaps_list), len(feats)),
                                 dtype=np.float32)
        feats_all[f, :] = feats

if args.layers == 'all':
    fmaps_train[layer_names[0]] = feats_all
elif args.layers == 'single':
    for l, dnn_layer in enumerate(layer_names):
        fmaps_train[dnn_layer] = np.squeeze(np.asarray(feats[l]))

# Standardize the data
scaler = []
for l, dnn_layer in enumerate(layer_names):
    scaler.append(StandardScaler())
    scaler[l].fit(fmaps_train[dnn_layer])
    fmaps_train[dnn_layer] = scaler[l].transform(fmaps_train[dnn_layer])

# Apply PCA
pca = []
for l, dnn_layer in enumerate(layer_names):
    pca.append(KernelPCA(n_components=args.n_components, kernel='poly',
                         degree=4, random_state=seed))
    pca[l].fit(fmaps_train[dnn_layer])
    fmaps_train[dnn_layer] = pca[l].transform(fmaps_train[dnn_layer])

del fmaps_train

save_dir = os.path.join(args.project_dir, 'dnn_feature_maps', 'pca')
if not op.exists(save_dir):
    os.makedirs(save_dir)

# Save scaler
with open(os.path.join(save_dir, args.dnn + '_' + str(args.pretrained)
                       + '_scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f, protocol=4)

# Save PCA
with open(os.path.join(save_dir, args.dnn + '_' + str(args.pretrained)
                       + '_pca.pkl'), 'wb') as f:
    pickle.dump(pca, f, protocol=4)


# =============================================================================
# Apply PCA on the ILSVRC-2012 validation images feature maps
# =============================================================================
# PCA is applied to partitions of 10k images feature maps for memory efficiency.
