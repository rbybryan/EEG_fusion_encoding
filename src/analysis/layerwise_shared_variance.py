"""Compute layer-wise SHARED variance (DNN ∩ LLM) for the rebuttal.

Companion to src/summarize_unique_r2_rebuttal.py.

Definition (standard 3-way Borcard-Legendre partition at the synthetic-EEG
prediction level, treating the fusion-trained model's prediction as the
"combined" predictor):

  X1 = DNN-only-pred at layer L      (synthetic EEG from CORnet-S layer L only)
  X2 = LLM-only-pred                 (synthetic EEG from text-embedding-large only)
  X12 = fusion-pred at layer L       (synthetic EEG from CORnet-S layer L + TEL)

  R^2(X1)  = R^2( y ~ DNN-only-pred-L )
  R^2(X2)  = R^2( y ~ LLM-only-pred )
  R^2(X12) = R^2( y ~ fusion-pred-L )

  SV(X1, X2) = R^2(X1) + R^2(X2) - R^2(X12)
  UV_std(LLM | DNN_L) = R^2(X12) - R^2(X1)
  UV_std(DNN_L | LLM) = R^2(X12) - R^2(X2)

All three quantities sum to R^2(X12): conserved 3-way decomposition.

Note: the existing layer-wise UV figures in rebuttal/figures/delta_r2_layerwise.png
use a slightly different UV definition (fusion-pred added on top of the
single-predictor model, i.e. a 2-predictor regression), which is an upper
bound on the standard UV. We report the standard SV here for the cleanest
3-way decomposition, and also save the standard UV curves so they can be
compared with the existing-definition curves.

Split-half resampling: matches src/summarize_unique_r2_rebuttal.py exactly
(seed=20260510 + subject id, 100 iterations, half of presentations averaged).

Outputs:
  - analysis/shared_variance_layerwise_summary.csv
  - analysis/shared_variance_layerwise_arrays.npz
  - rebuttal/figures/delta_r2_layerwise_shared.png  (SV-only, 5 layers)
  - rebuttal/figures/delta_r2_layerwise_3way.png    (UV_LLM | UV_DNN | SV)
"""

import argparse
import csv
import os
import os.path as op
import sys

import matplotlib

# --- path configuration (portable defaults for public release) ---
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_DATA_ROOT = _os.environ.get('EEG_FUSION_DATA', _os.path.join(_REPO, 'data'))
# --- end path configuration ---

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Avoid importing summarize_unique_r2_rebuttal directly because it pulls in
# sklearn, which is binary-incompatible with the local numpy 2.4.5.  Pull
# only the constants and small helpers from the sibling modules below.

PROJECT_DIR = _DATA_ROOT
OUT_DIR = _os.path.join(_REPO, 'rebuttal/figures')
ANALYSIS_DIR = _os.path.join(_REPO, 'analysis')

LAYERS = ['V1', 'V2', 'V4', 'IT', 'decoder']
SUBJECTS = list(range(1, 11))


def load_synth(project_dir, sub, name):
    base = op.join(project_dir, 'linear_results', f'sub-{sub:02d}',
                   'synthetic_eeg_data')
    for cand in (f'{name}.npy', f'{name}_all.npy'):
        path = op.join(base, cand)
        if op.exists(path):
            return np.load(path, allow_pickle=True).item()['synthetic_data']
    raise FileNotFoundError(f'no synth file for {name} under {base}')


def load_bio(project_dir, sub):
    path = op.join(project_dir, 'eeg_dataset', 'preprocessed_eeg_data_v1',
                   f'eeg_sub-{sub:02d}_split-test.npy')
    d = np.load(path, allow_pickle=True).item()
    return d['preprocessed_eeg_data'], np.asarray(d['times'])


def adjusted_r2(r2, n_obs, n_predictors):
    """Same adjusted-R^2 convention used in summarize_unique_r2_rebuttal.py."""
    denom = n_obs - n_predictors - 1
    if denom <= 0:
        return np.full_like(r2, np.nan, dtype=float)
    return 1.0 - (((1.0 - r2) * (n_obs - 1)) / denom)


def resample_indices(n, n_keep, seed):
    """sklearn.utils.resample(np.arange(n), replace=False)[:n_keep] equivalent.

    Uses legacy np.random.RandomState.permutation so the draw sequence
    matches sklearn's resample under the same integer seed.
    """
    return np.random.RandomState(seed).permutation(n)[:n_keep]


def single_predictor_r2_for_half(pred, bio_half):
    """R^2 of y ~ pred per channel x time. Returns raw and adjusted arrays.

    Vectorized: each channel/timepoint has its own scalar predictor across
    the 200 test images. Matches the centering/clip convention in
    summarize_unique_r2_rebuttal.nested_r2_for_half.
    """
    pred = pred[:, :63, :].reshape(pred.shape[0], -1).astype(float)
    y = bio_half[:, :63, :].reshape(bio_half.shape[0], -1).astype(float)

    pred = pred - pred.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    syy = np.sum(y * y, axis=0)
    spp = np.sum(pred * pred, axis=0)
    syp = np.sum(y * pred, axis=0)

    eps = np.finfo(float).eps
    valid = (syy > eps) & (spp > eps)
    r2 = np.zeros_like(syy)
    r2[valid] = (syp[valid] ** 2) / (syy[valid] * spp[valid])
    r2 = np.nan_to_num(np.clip(r2, 0.0, 1.0))

    r2_adj = adjusted_r2(r2, y.shape[0], 1)
    return r2.reshape(63, -1), r2_adj.reshape(63, -1)


def compute_subject_layer(project_dir, sub, layer, iterations, seed):
    dnn_name = f'cornet_s_r2_v3_layerwise_{layer}'
    fusion_name = f'cornet_s_with_text_embedding_large_r2_v3_layerwise_{layer}'
    llm_name = 'text_embedding_large_r2_v3'

    dnn_pred = load_synth(project_dir, sub, dnn_name)
    fusion_pred = load_synth(project_dir, sub, fusion_name)
    llm_pred = load_synth(project_dir, sub, llm_name)
    bio, times = load_bio(project_dir, sub)
    bio = bio[:, :, :63, :]

    rng = np.random.default_rng(seed + sub)
    r2_d, r2_d_a = [], []
    r2_l, r2_l_a = [], []
    r2_f, r2_f_a = [], []
    n_reps = bio.shape[1]
    for _ in range(iterations):
        seed_i = int(rng.integers(0, np.iinfo(np.int32).max))
        idx = resample_indices(n_reps, n_reps // 2, seed_i)
        bio_half = np.mean(np.delete(bio, idx, axis=1), axis=1)
        rd, rda = single_predictor_r2_for_half(dnn_pred, bio_half)
        rl, rla = single_predictor_r2_for_half(llm_pred, bio_half)
        rf, rfa = single_predictor_r2_for_half(fusion_pred, bio_half)
        r2_d.append(rd); r2_d_a.append(rda)
        r2_l.append(rl); r2_l_a.append(rla)
        r2_f.append(rf); r2_f_a.append(rfa)

    return {
        'r2_dnn': np.mean(r2_d, axis=0),
        'r2_dnn_adj': np.mean(r2_d_a, axis=0),
        'r2_llm': np.mean(r2_l, axis=0),
        'r2_llm_adj': np.mean(r2_l_a, axis=0),
        'r2_fusion': np.mean(r2_f, axis=0),
        'r2_fusion_adj': np.mean(r2_f_a, axis=0),
        'times': times,
    }


def winmean(mat, times, lo, hi):
    mask = (times >= lo) & (times <= hi)
    if not np.any(mask):
        return float('nan')
    return float(mat[:, mask].mean())


def plot_layer_curves(curves, title, ylabel, out_path,
                      cmap_name='viridis', figsize=(8, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap(cmap_name)
    for i, (label, mat, times) in enumerate(curves):
        mean = mat.mean(axis=0)
        sem = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
        color = cmap(i / max(1, len(curves) - 1))
        ax.plot(times, mean, color=color, label=label, lw=1.6)
        ax.fill_between(times, mean - sem, mean + sem,
                        color=color, alpha=0.18, lw=0)
    ax.axvline(0, color='k', lw=0.6, ls='--', alpha=0.6)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    if curves:
        ax.set_xlim(curves[0][2][0], curves[0][2][-1])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', default=PROJECT_DIR)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20260510)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    arrays = {}
    rows = []
    sv_curves = []
    uv_llm_curves = []
    uv_dnn_curves = []
    fusion_curves = []
    dnn_curves = []
    llm_curves = []

    for layer in LAYERS:
        print(f'layer {layer}', flush=True)
        subj_dnn = []; subj_dnn_adj = []
        subj_llm = []; subj_llm_adj = []
        subj_fusion = []; subj_fusion_adj = []
        times = None
        for sub in SUBJECTS:
            out = compute_subject_layer(
                args.project_dir, sub, layer,
                args.iterations, args.seed,
            )
            subj_dnn.append(out['r2_dnn']);       subj_dnn_adj.append(out['r2_dnn_adj'])
            subj_llm.append(out['r2_llm']);       subj_llm_adj.append(out['r2_llm_adj'])
            subj_fusion.append(out['r2_fusion']); subj_fusion_adj.append(out['r2_fusion_adj'])
            if times is None:
                times = out['times']

        r2_dnn = np.stack(subj_dnn);           r2_dnn_adj = np.stack(subj_dnn_adj)
        r2_llm = np.stack(subj_llm);           r2_llm_adj = np.stack(subj_llm_adj)
        r2_fusion = np.stack(subj_fusion);     r2_fusion_adj = np.stack(subj_fusion_adj)

        sv = r2_dnn + r2_llm - r2_fusion
        sv_adj = r2_dnn_adj + r2_llm_adj - r2_fusion_adj
        uv_llm_std = r2_fusion - r2_dnn
        uv_dnn_std = r2_fusion - r2_llm
        uv_llm_std_adj = r2_fusion_adj - r2_dnn_adj
        uv_dnn_std_adj = r2_fusion_adj - r2_llm_adj

        key = f'layerwise_shared__{layer}'
        arrays[f'{key}__r2_dnn']            = r2_dnn
        arrays[f'{key}__r2_llm']            = r2_llm
        arrays[f'{key}__r2_fusion']         = r2_fusion
        arrays[f'{key}__r2_dnn_adj']        = r2_dnn_adj
        arrays[f'{key}__r2_llm_adj']        = r2_llm_adj
        arrays[f'{key}__r2_fusion_adj']     = r2_fusion_adj
        arrays[f'{key}__sv']                = sv
        arrays[f'{key}__sv_adj']            = sv_adj
        arrays[f'{key}__uv_llm_std']        = uv_llm_std
        arrays[f'{key}__uv_dnn_std']        = uv_dnn_std
        arrays[f'{key}__uv_llm_std_adj']    = uv_llm_std_adj
        arrays[f'{key}__uv_dnn_std_adj']    = uv_dnn_std_adj
        arrays[f'{key}__times']             = times

        # Channel-mean curves per subject for plotting/summary.
        sv_curve     = sv_adj.mean(axis=1)            # (subj, time)
        uv_llm_curve = uv_llm_std_adj.mean(axis=1)
        uv_dnn_curve = uv_dnn_std_adj.mean(axis=1)
        dnn_curve    = r2_dnn_adj.mean(axis=1)
        llm_curve    = r2_llm_adj.mean(axis=1)
        fusion_curve = r2_fusion_adj.mean(axis=1)

        mean_sv = sv_curve.mean(axis=0)
        peak_idx = int(np.argmax(mean_sv))
        rows.append({
            'layer': layer,
            'predictor_fusion':  f'cornet_s_with_text_embedding_large_r2_v3_layerwise_{layer}',
            'predictor_dnn_only': f'cornet_s_r2_v3_layerwise_{layer}',
            'predictor_llm_only': 'text_embedding_large_r2_v3',
            'n_subjects':            len(SUBJECTS),
            'mean_sv_adj_all_time':  f'{float(sv_curve.mean()):.8f}',
            'peak_sv_adj':           f'{float(mean_sv[peak_idx]):.8f}',
            'peak_time_s':           f'{float(times[peak_idx]):.6f}',
            'sem_sv_at_peak':        f'{float(sv_curve[:, peak_idx].std(ddof=1) / np.sqrt(sv_curve.shape[0])):.8f}',
            'mean_sv_adj_0_200ms':   f'{winmean(sv_curve,     times, 0.0, 0.2):.8f}',
            'mean_sv_adj_200_600ms': f'{winmean(sv_curve,     times, 0.2, 0.6):.8f}',
            'mean_r2_dnn_only_peak': f'{float(dnn_curve.mean(axis=0).max()):.8f}',
            'mean_r2_llm_only_peak': f'{float(llm_curve.mean(axis=0).max()):.8f}',
            'mean_r2_fusion_peak':   f'{float(fusion_curve.mean(axis=0).max()):.8f}',
            'mean_uv_llm_std_peak':  f'{float(uv_llm_curve.mean(axis=0).max()):.8f}',
            'mean_uv_dnn_std_peak':  f'{float(uv_dnn_curve.mean(axis=0).max()):.8f}',
        })

        sv_curves.append((layer, sv_curve, times))
        uv_llm_curves.append((layer, uv_llm_curve, times))
        uv_dnn_curves.append((layer, uv_dnn_curve, times))
        fusion_curves.append((layer, fusion_curve, times))
        dnn_curves.append((layer, dnn_curve, times))
        llm_curves.append((layer, llm_curve, times))

    npz_path = op.join(ANALYSIS_DIR, 'shared_variance_layerwise_arrays.npz')
    np.savez_compressed(npz_path, **arrays)

    csv_path = op.join(ANALYSIS_DIR, 'shared_variance_layerwise_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Standalone SV plot.
    plot_layer_curves(
        sv_curves,
        'Layer-wise shared variance: DNN ∩ LLM (3-way partition)',
        'Adjusted shared R² = R²(DNN) + R²(LLM) − R²(fusion)',
        op.join(OUT_DIR, 'delta_r2_layerwise_shared.png'),
    )

    # 3-panel: UV(LLM, standard) | UV(DNN, standard) | SV.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    cmap = plt.get_cmap('viridis')
    for ax, curves, title in (
        (axes[0], uv_llm_curves, 'Layer-wise LLM unique  (standard: R²(fusion) − R²(DNN-only))'),
        (axes[1], uv_dnn_curves, 'Layer-wise DNN-layer unique  (standard: R²(fusion) − R²(LLM-only))'),
        (axes[2], sv_curves,     'Layer-wise shared variance  (R²(DNN) + R²(LLM) − R²(fusion))'),
    ):
        for i, (label, mat, times) in enumerate(curves):
            mean = mat.mean(axis=0)
            sem = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
            color = cmap(i / max(1, len(curves) - 1))
            ax.plot(times, mean, color=color, label=label, lw=1.6)
            ax.fill_between(times, mean - sem, mean + sem,
                            color=color, alpha=0.18, lw=0)
        ax.axvline(0, color='k', lw=0.6, ls='--', alpha=0.6)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel('Time (s)')
        ax.set_title(title, fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        if curves:
            ax.set_xlim(curves[0][2][0], curves[0][2][-1])
    axes[0].set_ylabel('Adjusted R² (channel mean)')
    fig.tight_layout()
    fig.savefig(op.join(OUT_DIR, 'delta_r2_layerwise_3way.png'), dpi=200)
    plt.close(fig)

    print(f'wrote {npz_path}')
    print(f'wrote {csv_path}')
    print(op.join(OUT_DIR, 'delta_r2_layerwise_shared.png'))
    print(op.join(OUT_DIR, 'delta_r2_layerwise_3way.png'))


if __name__ == '__main__':
    main()
