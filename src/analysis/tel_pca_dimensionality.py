"""Report PCA dimensionality + variance preserved for TEL (text-embedding-large).

TEL features = OpenAI text-embedding-large embeddings of GPT-4 captions,
averaged over 5 caption variants (`gpt4_features_embedded_5v_large_avg`).

Pre-PCA dimensionality: 3072 (OpenAI text-embedding-3-large native dim).
Manuscript PCA variants used: pc500 (rebuttal), pc1000 (legacy headline), pc3000.

Method:
  - Linear PCA via dual-form (compute X_c X_c^T eigvals, n_train x n_train);
    cheap because n_train (16495) << n_features (3072) -> rank-bounded by min(n).
  - For the actual manuscript baseline (legacy pc1000) the encoding pipeline
    used the `_pca.npy` file (linear PCA), so reporting linear-PCA EVR is the
    right metric for this LLM.

Outputs:
  - analysis/pca_dimensionality_tel.csv
"""

import csv
import os
import os.path as op

import numpy as np

# --- path configuration (portable defaults for public release) ---
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_DATA_ROOT = _os.environ.get('EEG_FUSION_DATA', _os.path.join(_REPO, 'data'))
# --- end path configuration ---


ANALYSIS_DIR = _os.path.join(_REPO, 'analysis')
TEL_PRE_PCA  = _os.path.join(_DATA_ROOT, 'gpt4_features/gpt4_features_embedded_5v_large_avg.npy')

os.makedirs(ANALYSIS_DIR, exist_ok=True)

d = np.load(TEL_PRE_PCA, allow_pickle=True).item()
X = d['text_features_train_long'].astype(np.float64)
n_train, n_feat = X.shape
print(f'TEL pre-PCA shape: ({n_train}, {n_feat})')

# Center.
mu = X.mean(axis=0)
Xc = X - mu

# Linear PCA via SVD on the dual form: G = Xc @ Xc.T is (n_train, n_train).
print('computing centered Gram matrix...')
G = Xc @ Xc.T
print('eigendecomposition...')
eigvals = np.linalg.eigvalsh(G)
eigvals = np.sort(eigvals)[::-1]
eigvals = np.maximum(eigvals, 0)

total = eigvals.sum()
csum = np.cumsum(eigvals)
print(f'n_nonzero eigvals = {(eigvals > 1e-6).sum()}')
print(f'top-1 EVR     = {eigvals[0]/total:.6f}')
print(f'top-100 EVR   = {csum[99]/total:.6f}')
print(f'top-500 EVR   = {csum[499]/total:.6f}')
print(f'top-1000 EVR  = {csum[999]/total:.6f}')
print(f'top-3000 EVR  = {csum[min(2999, len(eigvals)-1)]/total:.6f}')

rows = [{
    'model': 'TEL (OpenAI text-embedding-large of GPT-4 captions, 5v avg)',
    'pca_type': 'linear PCA (dual form, sklearn-equivalent)',
    'n_features_in': n_feat,
    'n_train_samples': n_train,
    'n_nonzero_eigvals': int((eigvals > 1e-6).sum()),
    'evr_top100':  f'{csum[99]/total:.6f}',
    'evr_top500':  f'{csum[499]/total:.6f}',
    'evr_top1000': f'{csum[999]/total:.6f}',
    'evr_top3000': f'{csum[min(2999, len(eigvals)-1)]/total:.6f}',
    'total_variance': f'{float(total/(n_train-1)):.4f}',
    'pre_pca_dim_source': 'OpenAI text-embedding-large native dim = 3072',
}]

csv_path = op.join(ANALYSIS_DIR, 'pca_dimensionality_tel.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(f'wrote {csv_path}')

# Also save the full eigenvalue vector for cumulative-EVR plotting later.
np.savez_compressed(
    op.join(ANALYSIS_DIR, 'pca_dimensionality_tel_eigvals.npz'),
    eigvals=eigvals, n_train=n_train, n_features=n_feat,
)
print(f'wrote {op.join(ANALYSIS_DIR, "pca_dimensionality_tel_eigvals.npz")}')
