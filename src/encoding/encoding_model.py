"""Predict the EEG responses using features from DNN and language models.

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
metric : str
    Regression metric to optimize ('r2', 'mse', or 'mae').
vision_only : bool
    If True, run image-only encoding.
language_only : bool
    If True, run text-only encoding.
fusion : bool
    If True, run joint image-text fusion encoding.
partition : bool
    If True, run the script on separate CPUs in parallel.
n_split : int
    Number of parallel splits for time slicing.
n : int
    Zero-based index of the current split to process.

Notes
-----
All (channel, timepoint) cells are solved together as columns of a single target
matrix. The batched solvers (``grid_search_batched`` / ``grid_search_fusion_batched``
in utils.py) reproduce the per-cell ``Grid_search`` / ``Grid_search_fusion``
selection and predictions exactly (same alpha grid, same KFold(5, shuffle,
random_state=42) CV, per-target alpha, and the convex-combination
train-scaled / test-unscaled fusion of ``WeightedRidge``), so partitioning the
time axis is now optional rather than necessary for tractability.
"""

import argparse
import os
import os.path as op
import random

import numpy as np

from utils import grid_search_batched, grid_search_fusion_batched


def str2bool(value):
    """Parse a string into a boolean for use as an argparse type.

    Parameters
    ----------
    value : bool or str
        Value to interpret as a boolean.

    Returns
    -------
    bool
        The parsed boolean value.

    Raises
    ------
    argparse.ArgumentTypeError
        If the value cannot be interpreted as a boolean.
    """
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {'true', '1', 'yes', 'y'}:
        return True
    if value in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Expected a boolean value, got {value!r}')


parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, default=4, help='Subject identifier')
parser.add_argument('--tot_eeg_chan', type=int, default=63,
                    help='Total number of EEG channels')
parser.add_argument('--tot_eeg_time', type=int, default=180,
                    help='Total number of EEG time points per trial')
parser.add_argument('--vision_model', type=str, default='cornet_s',
                    help='Vision model name')
parser.add_argument('--vision_pretrained', type=str2bool, default=True,
                    help='Whether to use pretrained vision PCA features')
parser.add_argument('--language_model', type=str, default='text_embedding_large',
                    help='Language model name')
parser.add_argument('--nfeature_vm', type=int, default=1000,
                    help='Number of vision feature components')
parser.add_argument('--nfeature_lm', type=int, default=1000,
                    help='Number of language feature components')
parser.add_argument('--project_dir', type=str,
                    default=os.environ.get('EEG_FUSION_DATA', 'data'),
                    help='Root project directory')
parser.add_argument('--pca_dir', type=str, default=None,
                    help='Directory containing pca_feature_maps; defaults to project_dir')
parser.add_argument('--tag', type=str, default='v3', help='Version tag for output files')
parser.add_argument('--metric', type=str, choices=['r2', 'mse', 'mae'], default='r2',
                    help='Regression metric to optimize')
parser.add_argument('--vision_only', action='store_true', help='Run image-only encoding')
parser.add_argument('--language_only', action='store_true', help='Run text-only encoding')
parser.add_argument('--fusion', action='store_true', help='Run image-text fusion encoding')
parser.add_argument('--partition', action='store_true', help='Run the script on separate CPUs')
parser.add_argument('--n_split', type=int, default=90, help='Number of parallel time splits')
parser.add_argument('--n', type=int, default=0, help='Zero-based split index to process')
parser.add_argument('--score_dtype', type=str, choices=['float64', 'float32'], default='float64',
                    help='CV scoring precision. float64 reproduces the published selection '
                         'exactly; float32 is ~1.7x faster with identical selection in our '
                         'tests. Final predictions are always float64.')
args = parser.parse_args()
score_dtype = np.float64 if args.score_dtype == 'float64' else np.float32

print('>>> Ridge encoding <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


random.seed(42)
np.random.seed(42)

pca_dir = args.pca_dir if args.pca_dir is not None else args.project_dir
vision_pretrained_dir = f'pretrained-{args.vision_pretrained}'
vision_output_name = (
    args.vision_model if args.vision_pretrained else f'{args.vision_model}_untrained'
)

embedding_file = 'gpt4_features_embedded_by_%s_pca.npy' % (args.language_model)
print(embedding_file)


### Loading the embeded text features ###
data_dict = np.load(op.join(args.project_dir, 'gpt4_features', embedding_file),
                    allow_pickle=True).item()
train_index = data_dict['train_index']
# Handle both key naming conventions
if 'embedding_train' in data_dict:
    language_embedding_train = data_dict['embedding_train'].astype(np.float32)
    language_embedding_test = data_dict['embedding_test'].astype(np.float32)
else:
    language_embedding_train = data_dict['text_features_train_long'].astype(np.float32)
    language_embedding_test = data_dict['text_features_test_long'].astype(np.float32)

# Average across five versions if not already averaged
if language_embedding_train.shape[0] == 5:
    language_embedding_train = np.average(language_embedding_train, axis=0)
    language_embedding_test = np.average(language_embedding_test, axis=0)
# Take the first n principal components for the language embeddings
n = language_embedding_train.shape[-1]
if n > args.nfeature_lm:
    language_embedding_train = language_embedding_train[:, :args.nfeature_lm]
    language_embedding_test = language_embedding_test[:, :args.nfeature_lm]

channels = 'preprocessed_eeg_data_v1'


### Loading the EEG training data ###
data_dir = os.path.join('eeg_dataset', channels)
training_file = 'eeg_' + 'sub-' + format(args.sub, '02') + '_split-train.npy'

data = np.load(os.path.join(args.project_dir, data_dir, training_file),
               allow_pickle=True).item()
y_train = data['preprocessed_eeg_data'].astype(np.float32)
ch_names = data['ch_names']
times = data['times']
y_train = y_train[train_index]

# Averaging across repetitions
y_train = np.mean(y_train, 1)[:, :args.tot_eeg_chan]

### Loading the EEG test data ###
test_file = 'eeg_' + 'sub-' + format(args.sub, '02') + '_split-test.npy'
data = np.load(os.path.join(args.project_dir, data_dir, test_file),
               allow_pickle=True).item()
y_test = data['preprocessed_eeg_data'].astype(np.float32)
# Averaging across repetitions
y_test = np.mean(y_test, 1)
y_test = y_test[:, :args.tot_eeg_chan]


feature_path = op.join(
    pca_dir,
    'pca_feature_maps',
    args.vision_model,
    vision_pretrained_dir,
    'layers-all',
    'pca_feature_maps_training.npy',
)
if not op.exists(feature_path):
    raise FileNotFoundError(f'Missing vision training PCA features: {feature_path}')
vision_feature_train = np.load(
    feature_path, allow_pickle=True).item()['all_layers'].astype(np.float32)
feature_path = op.join(
    pca_dir,
    'pca_feature_maps',
    args.vision_model,
    vision_pretrained_dir,
    'layers-all',
    'pca_feature_maps_test.npy',
)
if not op.exists(feature_path):
    raise FileNotFoundError(f'Missing vision test PCA features: {feature_path}')
vision_feature_test = np.load(
    feature_path, allow_pickle=True).item()['all_layers'].astype(np.float32)

vision_feature_train = vision_feature_train[:, :args.nfeature_vm]
vision_feature_test = vision_feature_test[:, :args.nfeature_vm]
vision_feature_train = vision_feature_train[train_index]

# identify alpha ranges and convex weights to search over
alpha_range = np.logspace(-3, 7, 100)
weight_list = np.linspace(0, 1000, 41) / 1000

if args.partition:
    ## select the slices of data for distributing computation on different cpus
    args.tot_eeg_time = args.tot_eeg_time // args.n_split
    y_train = y_train[:, :, args.n * args.tot_eeg_time:(args.n + 1) * args.tot_eeg_time]
    y_test = y_test[:, :, args.n * args.tot_eeg_time:(args.n + 1) * args.tot_eeg_time]
    times = times[args.n * args.tot_eeg_time:(args.n + 1) * args.tot_eeg_time]
else:
    args.n = 'all'

chan = args.tot_eeg_chan
tp = args.tot_eeg_time
n_train = y_train.shape[0]
n_test = y_test.shape[0]

print("after slicing:", tp, y_train.shape, y_test.shape)

# Stack every (channel, timepoint) cell as a column -> solve all targets at once.
# Target column index is j = c * tp + t (reshape of (n, chan, tp) in C-order).
Y_train_2d = y_train.reshape(n_train, chan * tp)

save_dir = os.path.join(args.project_dir, 'linear_results', 'sub-' +
                        format(args.sub, '02'), 'synthetic_eeg_data')

if not op.exists(save_dir):
    os.makedirs(save_dir)


def _to_save_order(arr):
    """Per-target (j = c*tp + t) -> the original per-cell append order (t-major, c-minor)."""
    return list(np.asarray(arr).reshape(chan, tp).T.flatten())


# Vision-only encoding
if args.vision_only:
    Y_pred, best_alpha, _ = grid_search_batched(
        vision_feature_train, vision_feature_test, Y_train_2d,
        alpha_range=alpha_range, kfold=5, metric=args.metric, score_dtype=score_dtype)
    y_pred_vision_only = Y_pred.reshape(n_test, chan, tp)
    data_dict = {
        'synthetic_data': y_pred_vision_only,
        'ch_names': ch_names,
        'times': times,
        'alpha': _to_save_order(best_alpha),
    }
    file_name = '%s_%s_%s_%s.npy' % (vision_output_name, args.metric, args.tag, args.n)
    np.save(os.path.join(save_dir, file_name), data_dict)


# Language-only encoding
if args.language_only:
    Y_pred, best_alpha, _ = grid_search_batched(
        language_embedding_train, language_embedding_test, Y_train_2d,
        alpha_range=alpha_range, kfold=5, metric=args.metric, score_dtype=score_dtype)
    y_pred_language_only = Y_pred.reshape(n_test, chan, tp)
    data_dict = {
        'synthetic_data': y_pred_language_only,
        'ch_names': ch_names,
        'times': times,
        'alpha': _to_save_order(best_alpha),
    }
    file_name = '%s_%s_%s_%s.npy' % (args.language_model, args.metric, args.tag, args.n)
    np.save(os.path.join(save_dir, file_name), data_dict)


# Image-text fusion encoding (convex combination preserved exactly)
if args.fusion:
    Y_pred, best_alpha, best_weight, flag = grid_search_fusion_batched(
        vision_feature_train, vision_feature_test,
        language_embedding_train, language_embedding_test, Y_train_2d,
        alpha_range=alpha_range, weight_list=weight_list,
        kfold=5, metric=args.metric, score_dtype=score_dtype)
    y_pred_fusion = Y_pred.reshape(n_test, chan, tp)
    data_dict = {
        'synthetic_data': y_pred_fusion,
        'ch_names': ch_names,
        'times': times,
        'alpha': _to_save_order(best_alpha),
        'weight': _to_save_order(best_weight),
        'flag': _to_save_order(flag),
    }
    file_name = '%s_with_%s_%s_%s_%s.npy' % (
        vision_output_name, args.language_model, args.metric, args.tag, args.n)
    np.save(os.path.join(save_dir, file_name), data_dict)
