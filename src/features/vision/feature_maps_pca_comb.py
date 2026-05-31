"""PCA is performed on the DNN feature maps to reduce their dimensionality.
PCA is applied on either the feature maps of single DNN layers, or on the
appended feature maps of all layers.
Before applying PCA on the CORnet-S feature maps, run
'sort_feature_maps_cornet_s.py'.

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
import numpy as np
import os
import os.path as op
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--dnn', default='cornet_s', type=str)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--layers', default='all', type=str)
parser.add_argument('--n_components', default=10000, type=int)
parser.add_argument('--project_dir', default='/scratch/byrong/encoding/data', type=str)
args = parser.parse_args()

print('>>> Apply PCA on the feature maps <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220

text_embedding_file = 'gpt4_features_embedded_5v_large_avg_pca.npy'
data_dict = np.load(op.join(args.project_dir,'gpt4_features', text_embedding_file), allow_pickle=True).item()
text_features_train_long = data_dict["text_features_train_long"] 
train_index = data_dict["train_index"]
text_features_test_long = data_dict["text_features_test_long"]
# =============================================================================
# Apply PCA on the training images feature maps
# =============================================================================
# The standardization and PCA statistics computed on the training images feature
# maps are also applied to the test images feature maps and to the ILSVRC-2012
# images feature maps.

# Load the feature maps
feats = []
feats_all = []
fmaps_train = {}
fmaps_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'full_feature_maps', args.dnn, 'pretrained-'+str(args.pretrained),
	'training_images')
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
		feats_all.append(feats)
if args.layers == 'all':
	fmaps_train[layer_names[0]] = np.asarray(feats_all)
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
for l, dnn_layer in enumerate(layer_names):
	print(fmaps_train[dnn_layer].shape)
	fmaps_train[dnn_layer] = fmaps_train[dnn_layer][train_index]
	fmaps_train[dnn_layer] = np.concatenate([fmaps_train[dnn_layer],text_features_train_long],axis = -1)
	print(fmaps_train[dnn_layer].shape)

pca = []
for l, dnn_layer in enumerate(layer_names):
	pca.append(KernelPCA(n_components=args.n_components, kernel='poly',
		degree=4, random_state=seed))
	pca[l].fit(fmaps_train[dnn_layer])
	
	fmaps_train[dnn_layer] = pca[l].transform(fmaps_train[dnn_layer])

# Save the downsampled feature maps
save_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'pca_feature_maps', args.dnn+str(args.n_components), 'pretrained-'+str(args.pretrained), 'layers-'+
	args.layers)
file_name = 'pca_feature_maps_training_comb'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), fmaps_train)

# for l, dnn_layer in enumerate(layer_names):
# 	lambdas = np.var(fmaps_train[dnn_layer], axis=0)
# 	explained_variance_ratio = lambdas / np.sum(lambdas)

# 	print("Explained variance ratio:", explained_variance_ratio)

for l, dnn_layer in enumerate(layer_names):
	
	# Compute explained variance
	eigenvalues = pca[l].eigenvalues_  # Eigenvalues of the kernel matrix
	explained_variances = eigenvalues / np.sum(eigenvalues)
	np.save(os.path.join(save_dir, 'explained_variance_%d.npy'%l),explained_variances)
np.save(os.path.join(save_dir, 'pca.npy'%l),pca)

del fmaps_train


# =============================================================================
# Apply PCA on the test images feature maps
# =============================================================================
# Load the feature maps
feats = []
feats_all = []
fmaps_test = {}
fmaps_dir = os.path.join(args.project_dir, 'dnn_feature_maps',
	'full_feature_maps', args.dnn, 'pretrained-'+str(args.pretrained),
	'test_images')
fmaps_list = os.listdir(fmaps_dir)
fmaps_list.sort()
for f, fmaps in enumerate(fmaps_list):
	fmaps_data = np.load(os.path.join(fmaps_dir, fmaps),
		allow_pickle=True).item()
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
		feats_all.append(feats)
if args.layers == 'all':
	fmaps_test[layer_names[0]] = np.asarray(feats_all)
elif args.layers == 'single':
	for l, dnn_layer in enumerate(layer_names):
		fmaps_test[dnn_layer] = np.squeeze(np.asarray(feats[l]))

# Standardize the data
for l, dnn_layer in enumerate(layer_names):
	fmaps_test[dnn_layer] = scaler[l].transform(fmaps_test[dnn_layer])

for l, dnn_layer in enumerate(layer_names):
	print(fmaps_test[dnn_layer].shape)
	fmaps_test[dnn_layer] = np.concatenate([fmaps_test[dnn_layer],text_features_test_long],axis = -1)
	print(fmaps_test[dnn_layer].shape)


# Apply PCA
for l, dnn_layer in enumerate(layer_names):
	fmaps_test[dnn_layer] = pca[l].transform(fmaps_test[dnn_layer])

# Save the downsampled feature maps
file_name = 'pca_feature_maps_test_comb'
np.save(os.path.join(save_dir, file_name), fmaps_test)
del fmaps_test


# =============================================================================
# Apply PCA on the ILSVRC-2012 validation images feature maps
# =============================================================================
# PCA is applied to partitions of 10k images feature maps for memory efficiency.
