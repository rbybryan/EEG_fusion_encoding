"""Compute pairwise cosine similarity between DNN layer features and LLM embeddings.

Adapted from similarity-between-model.ipynb.

Analyses:
  1b. Overall cosine similarity between DNN (pretrained & untrained) and LLM
      representations — reports Pearson r between lower-triangle RDM vectors.
  1c. Layer-by-layer cosine similarity between individual DNN layers and LLM
      representations — tracks representational similarity across feature
      complexity.

Usage
-----
# 1b: Overall DNN vs LLM similarity (pretrained and untrained)
python similarity_dnn_llm.py --mode overall

# 1c: Layer-by-layer DNN vs LLM similarity
python similarity_dnn_llm.py --mode layerwise

# Both
python similarity_dnn_llm.py --mode all
"""

import argparse
import os
import os.path as op
import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
import pickle
import time

# --- path configuration (portable defaults for public release) ---
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_DATA_ROOT = _os.environ.get('EEG_FUSION_DATA', _os.path.join(_REPO, 'data'))
# --- end path configuration ---


# =============================================================================
# Arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='all',
                    choices=['overall', 'layerwise', 'all'],
                    help='Which analysis to run')
parser.add_argument('--dnn', type=str, default='cornet_s',
                    choices=['cornet_s', 'alexnet', 'resnet50', 'moco',
                             'convnext_xlarge', 'tf_efficientnetv2_l'],
                    help='DNN model to use')
parser.add_argument('--n_components', type=int, default=1000,
                    help='Number of PCA components for feature reduction')
parser.add_argument('--project_dir', type=str,
                    default=_DATA_ROOT,
                    help='Project data directory')
parser.add_argument('--text_embedding_file', type=str,
                    default='gpt4_features_embedded_5v_large_cleaned_avg_pca.npy',
                    help='LLM embedding file name')
args = parser.parse_args()


# =============================================================================
# Layer names per DNN
# =============================================================================
DNN_LAYERS = {
    'cornet_s': ['V1', 'V2', 'V4', 'IT', 'decoder'],
    'alexnet': ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'decoder'],
    'resnet50': ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool'],
    'moco': ['layer1', 'layer2', 'layer3', 'layer4', 'avgpool'],
    'convnext_xlarge': ['stages.0', 'stages.1', 'stages.2', 'stages.3', 'decoder'],
    'tf_efficientnetv2_l': ['blocks.0.3', 'blocks.1.6', 'blocks.2.6',
                            'blocks.4.18', 'blocks.6.6', 'decoder'],
}


# =============================================================================
# Helper functions
# =============================================================================
def load_llm_embeddings(project_dir, text_embedding_file):
    """Load LLM embeddings and training index."""
    fpath = op.join(project_dir, 'gpt4_features', text_embedding_file)
    data_dict = np.load(fpath, allow_pickle=True).item()
    train_index = data_dict['train_index'].copy()
    if 'embedding_train' in data_dict:
        text_train = data_dict['embedding_train'].copy()
    else:
        text_train = data_dict['text_features_train_long'].copy()
    return text_train, train_index


def load_pca_features(project_dir, dnn, pretrained, layers='all'):
    """Load PCA-transformed features from pca_feature_maps/.

    Returns dict: {layer_name: np.ndarray of shape (n_images, n_pca_components)}
    """
    pca_path = op.join(project_dir, 'pca_feature_maps', dnn,
                       f'pretrained-{pretrained}', f'layers-{layers}',
                       'pca_feature_maps_training.npy')
    if op.isfile(pca_path):
        data = np.load(pca_path, allow_pickle=True).item()
        print(f'  Loaded PCA features from {pca_path}')
        return data
    return None


def load_full_feature_maps(project_dir, dnn, pretrained, partition='training_images'):
    """Load per-image full feature maps and stack them.

    Returns dict: {layer_name: np.ndarray of shape (n_images, n_features_flat)}
    """
    fmaps_dir = op.join(project_dir, 'dnn_feature_maps', 'full_feature_maps',
                        dnn, f'pretrained-{pretrained}', partition)
    if not op.isdir(fmaps_dir):
        raise FileNotFoundError(
            f"Feature maps directory not found: {fmaps_dir}\n"
            f"Run the extraction script first:\n"
            f"  python extract_feature_maps_{dnn}.py --pretrained {pretrained}")

    fmaps_list = sorted(os.listdir(fmaps_dir))
    fmaps_list = [f for f in fmaps_list if f.endswith('.npy')]

    # First pass: determine layer names
    sample = np.load(op.join(fmaps_dir, fmaps_list[0]), allow_pickle=True).item()
    layer_names = list(sample.keys())

    # Second pass: collect all
    features_per_layer = {layer: [] for layer in layer_names}
    for fname in fmaps_list:
        fmap = np.load(op.join(fmaps_dir, fname), allow_pickle=True).item()
        for layer in layer_names:
            features_per_layer[layer].append(fmap[layer].flatten())

    for layer in layer_names:
        features_per_layer[layer] = np.stack(features_per_layer[layer], axis=0)

    return features_per_layer


def concat_all_layers(features_per_layer):
    """Concatenate all layer features into a single array."""
    return np.concatenate(list(features_per_layer.values()), axis=1)


def apply_pca(features, n_components, seed=20200220):
    """Standardize + Kernel PCA on features. Skip PCA if features already small."""
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    if features.shape[1] <= n_components:
        return features
    pca = KernelPCA(n_components=n_components, kernel='poly', degree=4,
                    random_state=seed)
    return pca.fit_transform(features)


def compute_rdm_correlation(features_a, features_b, metric='cosine'):
    """Compute Pearson r between lower-triangle RDM vectors.

    Parameters
    ----------
    features_a, features_b : ndarray, shape (n_images, n_features)
    metric : str, 'cosine' for cosine similarity based RDM

    Returns
    -------
    r : float, Pearson correlation
    p : float, p-value
    """
    sim_a = cosine_similarity(features_a)
    sim_b = cosine_similarity(features_b)
    lower_idx = np.tril_indices(sim_a.shape[0], -1)
    vec_a = sim_a[lower_idx]
    vec_b = sim_b[lower_idx]
    r, p = pearsonr(vec_a, vec_b)
    return r, p


# =============================================================================
# Main analyses
# =============================================================================
def run_overall(args):
    """1b: Compare overall DNN vs LLM similarity for pretrained and untrained."""
    print('=' * 60)
    print('1b: Overall DNN vs LLM representational similarity')
    print('=' * 60)

    text_train, train_index = load_llm_embeddings(
        args.project_dir, args.text_embedding_file)

    results = {}
    for pretrained in [True, False]:
        label = 'pretrained' if pretrained else 'untrained'
        print(f'\n--- {args.dnn} ({label}) ---')

        # Try PCA features first (much faster, less memory)
        pca_data = load_pca_features(args.project_dir, args.dnn, pretrained,
                                     layers='all')
        if pca_data is not None:
            all_feats_pca = pca_data['all_layers'][train_index]
            all_feats_pca = all_feats_pca[:, :args.n_components]
            print(f'  Using pre-computed PCA features: {all_feats_pca.shape}')
        else:
            # Fall back to loading raw features + in-memory PCA
            try:
                features = load_full_feature_maps(
                    args.project_dir, args.dnn, pretrained)
            except FileNotFoundError as e:
                print(f'  SKIPPED: {e}')
                continue
            all_feats = concat_all_layers(features)
            all_feats = all_feats[train_index]
            print(f'  DNN features shape (all layers): {all_feats.shape}')
            all_feats_pca = apply_pca(all_feats, args.n_components)
            print(f'  After PCA: {all_feats_pca.shape}')

        # Compare with LLM
        r, p = compute_rdm_correlation(all_feats_pca, text_train)
        print(f'  Pearson r (DNN vs LLM cosine similarity): {r:.4f}, p = {p:.2e}')
        results[label] = {'r': r, 'p': p}

    # Save results
    save_dir = op.join(args.project_dir, 'RSA')
    os.makedirs(save_dir, exist_ok=True)
    save_path = op.join(save_dir, f'overall_similarity_{args.dnn}.npy')
    np.save(save_path, results)
    print(f'\nResults saved to {save_path}')
    return results


def run_layerwise(args):
    """1c: Layer-by-layer DNN vs LLM similarity."""
    print('=' * 60)
    print('1c: Layer-by-layer DNN vs LLM representational similarity')
    print('=' * 60)

    text_train, train_index = load_llm_embeddings(
        args.project_dir, args.text_embedding_file)

    results = {}
    for pretrained in [True, False]:
        label = 'pretrained' if pretrained else 'untrained'
        print(f'\n--- {args.dnn} ({label}) ---')

        # Try PCA features first (single-layer mode)
        pca_data = load_pca_features(args.project_dir, args.dnn, pretrained,
                                     layers='single')
        if pca_data is not None:
            layer_results = {}
            for layer_name, layer_feats in pca_data.items():
                layer_feats = layer_feats[train_index]
                n_comp = min(args.n_components, layer_feats.shape[1])
                layer_feats = layer_feats[:, :n_comp]
                r, p = compute_rdm_correlation(layer_feats, text_train)
                print(f'  {layer_name:>15s}: r = {r:.4f}, p = {p:.2e}  '
                      f'(n_pca={n_comp})')
                layer_results[layer_name] = {'r': r, 'p': p, 'n_pca': n_comp}
            results[label] = layer_results
        else:
            # Fall back to raw features
            try:
                features = load_full_feature_maps(
                    args.project_dir, args.dnn, pretrained)
            except FileNotFoundError as e:
                print(f'  SKIPPED: {e}')
                continue

            layer_results = {}
            for layer_name, layer_feats in features.items():
                layer_feats = layer_feats[train_index]
                n_comp = min(args.n_components, layer_feats.shape[1])
                layer_feats_pca = apply_pca(layer_feats, n_comp)
                r, p = compute_rdm_correlation(layer_feats_pca, text_train)
                print(f'  {layer_name:>15s}: r = {r:.4f}, p = {p:.2e}  '
                      f'(n_features={layer_feats.shape[1]}, n_pca={n_comp})')
                layer_results[layer_name] = {'r': r, 'p': p,
                                             'n_features': layer_feats.shape[1],
                                             'n_pca': n_comp}
            results[label] = layer_results

    # Save results
    save_dir = op.join(args.project_dir, 'RSA')
    os.makedirs(save_dir, exist_ok=True)
    save_path = op.join(save_dir, f'layerwise_similarity_{args.dnn}.npy')
    np.save(save_path, results)
    print(f'\nResults saved to {save_path}')
    return results


# =============================================================================
# Entry point
# =============================================================================
if __name__ == '__main__':
    t0 = time.time()
    if args.mode in ('overall', 'all'):
        run_overall(args)
    if args.mode in ('layerwise', 'all'):
        run_layerwise(args)
    print(f'\nTotal time: {time.time() - t0:.1f}s')
