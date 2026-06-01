"""Same-modality fusion control encoding model.

Runs encoding with two models of the same modality:
  - vision+vision  : e.g., cornet_s + resnet50
                     Features loaded from pca_feature_maps/{model}/pretrained-True/layers-all/
  - language+language: e.g., text_embedding_large + ember-v1
                     Features loaded from gpt4_features/gpt4_features_embedded_by_{model}_pca.npy

Supports model1_only, model2_only, and fusion modes.

Usage
-----
# Language+language, all three modes
python encoding_model_samemodality.py \\
    --sub 4 --modality language \\
    --model1 text_embedding_large --model2 ember-v1 \\
    --model1_only --model2_only --fusion \\
    --partition --n_split 90 --n $N --metric r2 --tag v3_ll

# Vision+vision, all three modes
python encoding_model_samemodality.py \\
    --sub 4 --modality vision \\
    --model1 cornet_s --model2 resnet50 \\
    --model1_only --model2_only --fusion \\
    --partition --n_split 90 --n $N --metric r2 --tag v3_vv
"""

import argparse
import os
import os.path as op
import numpy as np
import random
from utils import Grid_search, Grid_search_fusion

try:
    from batch_multiridge import batch_encoding, batch_fusion_encoding
    BATCH_AVAILABLE = True
except ImportError:
    BATCH_AVAILABLE = False


# =============================================================================
# Arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub',           type=int,   default=4)
parser.add_argument('--tot_eeg_chan',  type=int,   default=63)
parser.add_argument('--tot_eeg_time',  type=int,   default=180)
parser.add_argument('--modality',      type=str,   default='language',
                    choices=['vision', 'language'],
                    help='Both models must share the same modality')
parser.add_argument('--model1',        type=str,   default='text_embedding_large')
parser.add_argument('--model2',        type=str,   default='ember-v1')
parser.add_argument('--nfeature1',     type=int,   default=1000)
parser.add_argument('--nfeature2',     type=int,   default=1000)
parser.add_argument('--project_dir',   type=str,   default='/scratch/byrong/encoding/data')
parser.add_argument('--tag',           type=str,   default='v3_samemod')
parser.add_argument('--metric',        type=str,   default='r2', choices=['r2', 'mse', 'mae'])
parser.add_argument('--model1_only',   action='store_true')
parser.add_argument('--model2_only',   action='store_true')
parser.add_argument('--fusion',        action='store_true')
parser.add_argument('--partition',     action='store_true')
parser.add_argument('--n_split',       type=int,   default=90)
parser.add_argument('--n',             type=int,   default=0)
parser.add_argument('--use_batch_approach', action='store_true',
                    help='Use batch all-targets-together approach for speed')
args = parser.parse_args()

print('>>> Same-modality ridge encoding <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

random.seed(42)
np.random.seed(42)


# =============================================================================
# Load train_index from the reference language file (same split for all models)
# =============================================================================
_ref_path = op.join(args.project_dir, 'gpt4_features',
                    'gpt4_features_embedded_by_text_embedding_large_pca.npy')
_ref = np.load(_ref_path, allow_pickle=True).item()
train_index = _ref['train_index']


# =============================================================================
# Feature loaders
# =============================================================================
def load_vision_features(model_name, nfeature):
    """Load all-layers PCA features from pca_feature_maps/ and index by train_index."""
    train_path = op.join(args.project_dir, 'pca_feature_maps', model_name,
                         'pretrained-True', 'layers-all', 'pca_feature_maps_training.npy')
    test_path  = op.join(args.project_dir, 'pca_feature_maps', model_name,
                         'pretrained-True', 'layers-all', 'pca_feature_maps_test.npy')
    feats_train = np.load(train_path, allow_pickle=True).item()['all_layers'].astype(np.float32)
    feats_test  = np.load(test_path,  allow_pickle=True).item()['all_layers'].astype(np.float32)
    feats_train = feats_train[train_index, :nfeature]
    feats_test  = feats_test[:, :nfeature]
    return feats_train, feats_test


def load_language_features(model_name, nfeature):
    """Load pre-indexed language embeddings from gpt4_features/."""
    fname = 'gpt4_features_embedded_by_%s_pca.npy' % model_name
    fpath = op.join(args.project_dir, 'gpt4_features', fname)
    d = np.load(fpath, allow_pickle=True).item()
    if 'embedding_train' in d:
        feats_train = d['embedding_train'].astype(np.float32)
        feats_test  = d['embedding_test'].astype(np.float32)
    else:
        feats_train = d['text_features_train_long'].astype(np.float32)
        feats_test  = d['text_features_test_long'].astype(np.float32)
    feats_train = feats_train[:, :nfeature]
    feats_test  = feats_test[:, :nfeature]
    return feats_train, feats_test


# =============================================================================
# Load features
# =============================================================================
if args.modality == 'vision':
    feats1_train, feats1_test = load_vision_features(args.model1, args.nfeature1)
    feats2_train, feats2_test = load_vision_features(args.model2, args.nfeature2)
else:
    feats1_train, feats1_test = load_language_features(args.model1, args.nfeature1)
    feats2_train, feats2_test = load_language_features(args.model2, args.nfeature2)

print('Model1 features: train %s  test %s' % (feats1_train.shape, feats1_test.shape))
print('Model2 features: train %s  test %s' % (feats2_train.shape, feats2_test.shape))


# =============================================================================
# Load EEG
# =============================================================================
channels = 'preprocessed_eeg_data_v1'
data_dir  = os.path.join('eeg_dataset', channels)

training_file = 'eeg_sub-%s_split-train.npy' % format(args.sub, '02')
data    = np.load(os.path.join(args.project_dir, data_dir, training_file),
                  allow_pickle=True).item()
y_train   = data['preprocessed_eeg_data'].astype(np.float32)
ch_names  = data['ch_names']
times     = data['times']
y_train   = y_train[train_index]
y_train   = np.mean(y_train, 1)[:, :args.tot_eeg_chan]

test_file = 'eeg_sub-%s_split-test.npy' % format(args.sub, '02')
data    = np.load(os.path.join(args.project_dir, data_dir, test_file),
                  allow_pickle=True).item()
y_test  = data['preprocessed_eeg_data'].astype(np.float32)
y_test  = np.mean(y_test, 1)[:, :args.tot_eeg_chan]


# =============================================================================
# Partition (time-split parallelisation)
# =============================================================================
alpha_range = np.logspace(-3, 7, 100)

if args.partition:
    args.tot_eeg_time = args.tot_eeg_time // args.n_split
    y_train = y_train[:, :, args.n * args.tot_eeg_time:(args.n + 1) * args.tot_eeg_time]
    y_test  = y_test[:,  :, args.n * args.tot_eeg_time:(args.n + 1) * args.tot_eeg_time]
    times   = times[args.n * args.tot_eeg_time:(args.n + 1) * args.tot_eeg_time]
else:
    args.n = 'all'

print('After slicing: tot_eeg_time=%d  y_train=%s  y_test=%s' %
      (args.tot_eeg_time, y_train.shape, y_test.shape))


# =============================================================================
# Encoding implementations
# =============================================================================
def batch_approach():
    results = {}
    if args.model1_only:
        predictions, best_alphas, _ = batch_encoding(
            feats1_train, feats1_test, y_train, y_test, alpha_range,
            metric=args.metric, cv_folds=5
        )
        results['model1'] = (predictions, best_alphas.flatten().tolist())

    if args.model2_only:
        predictions, best_alphas, _ = batch_encoding(
            feats2_train, feats2_test, y_train, y_test, alpha_range,
            metric=args.metric, cv_folds=5
        )
        results['model2'] = (predictions, best_alphas.flatten().tolist())

    if args.fusion:
        predictions, best_alphas, best_weights, _ = batch_fusion_encoding(
            feats1_train, feats1_test,
            feats2_train, feats2_test,
            y_train, y_test,
            alpha_range, np.linspace(0, 1000, 41) / 1000,
            metric=args.metric, cv_folds=5
        )
        results['fusion'] = (
            predictions,
            best_alphas.flatten().tolist(),
            best_weights.flatten().tolist(),
            [0] * best_alphas.size,
        )
    return results


def individual_approach():
    y_pred1 = np.zeros(y_test.shape)
    y_pred2 = np.zeros(y_test.shape)
    y_pred_fusion = np.zeros(y_test.shape)
    alpha1, alpha2, alpha_fus = [], [], []
    weight_fus, flag_fus = [], []

    for t in range(args.tot_eeg_time):
        for c in range(args.tot_eeg_chan):
            if args.model1_only:
                y_pred1[:, c, t], alpha, flag = Grid_search(
                    feats1_train, feats1_test,
                    y_train[:, c, t], y_test[:, c, t],
                    alpha_range=alpha_range, metric=args.metric, kfold=5)
                alpha1.append(alpha)

            if args.model2_only:
                y_pred2[:, c, t], alpha, flag = Grid_search(
                    feats2_train, feats2_test,
                    y_train[:, c, t], y_test[:, c, t],
                    alpha_range=alpha_range, metric=args.metric, kfold=5)
                alpha2.append(alpha)

            if args.fusion:
                y_pred_fusion[:, c, t], alpha, weight, flag = Grid_search_fusion(
                    feats1_train, feats1_test,
                    feats2_train, feats2_test,
                    y_train[:, c, t], y_test[:, c, t],
                    alpha_range=alpha_range,
                    weight_list=np.linspace(0, 1000, 41) / 1000,
                    metric=args.metric, kfold=5)
                alpha_fus.append(alpha)
                weight_fus.append(weight)
                flag_fus.append(flag)

    return {
        'model1': (y_pred1, alpha1),
        'model2': (y_pred2, alpha2),
        'fusion': (y_pred_fusion, alpha_fus, weight_fus, flag_fus),
    }


if args.use_batch_approach:
    if not BATCH_AVAILABLE:
        raise ImportError('batch_multiridge is unavailable; cannot use --use_batch_approach')
    results = batch_approach()
else:
    results = individual_approach()


# =============================================================================
# Save results
# =============================================================================
save_dir = os.path.join(args.project_dir, 'linear_results',
                        'sub-' + format(args.sub, '02'), 'synthetic_eeg_data')
os.makedirs(save_dir, exist_ok=True)

if args.model1_only:
    fname = '%s_%s_%s_%s.npy' % (args.model1, args.metric, args.tag, args.n)
    y_pred1, alpha1 = results['model1']
    np.save(op.join(save_dir, fname),
            {'synthetic_data': y_pred1, 'ch_names': ch_names,
             'times': times, 'alpha': alpha1})

if args.model2_only:
    fname = '%s_%s_%s_%s.npy' % (args.model2, args.metric, args.tag, args.n)
    y_pred2, alpha2 = results['model2']
    np.save(op.join(save_dir, fname),
            {'synthetic_data': y_pred2, 'ch_names': ch_names,
             'times': times, 'alpha': alpha2})

if args.fusion:
    fname = '%s_with_%s_%s_%s_%s.npy' % (
        args.model1, args.model2, args.metric, args.tag, args.n)
    y_pred_fusion, alpha_fus, weight_fus, flag_fus = results['fusion']
    np.save(op.join(save_dir, fname),
            {'synthetic_data': y_pred_fusion, 'ch_names': ch_names,
             'times': times, 'alpha': alpha_fus,
             'weight': weight_fus, 'flag': flag_fus})

print('Saved to %s' % save_dir)
