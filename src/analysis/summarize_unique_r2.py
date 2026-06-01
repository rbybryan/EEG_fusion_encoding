"""Compute true nested-model delta R2 for rebuttal variance partitioning.

This follows the variance-partitioning pattern in
analysis/variance_partition-24.03.25.ipynb: fit a reduced model, fit a
full nested model, and take full R2 - reduced R2. The notebook reports
adjusted R2 differences; this script writes both raw and adjusted delta
R2 and uses adjusted delta R2 for the headline figures.

For each contrast:
  reduced model: EEG ~ control_prediction
  full model:    EEG ~ control_prediction + predictor_prediction

The synthetic prediction files are already held-out test predictions
with shape (images, channels, time). The biological target follows the
same split-half resampling convention used by src/correlation.py and
src/partial_correlation.py.

Outputs:
  - analysis/delta_r2_rebuttal_summary.csv
  - analysis/delta_r2_rebuttal_arrays.npz
  - rebuttal/figures/delta_r2_trained_vs_untrained.png
  - rebuttal/figures/delta_r2_layerwise.png
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
from sklearn.utils import resample

sys.path.insert(0, op.dirname(op.abspath(__file__)))
from run_partial_corr import (  # noqa: E402
    LAYERS,
    SUBJECTS,
    TRAINED_VS_UNTRAINED,
)

PROJECT_DIR = _DATA_ROOT
OUT_DIR = _os.path.join(_REPO, 'rebuttal/figures')
ANALYSIS_DIR = _os.path.join(_REPO, 'analysis')

TRAINED_PRETTY = {
    'cornet_s_untrained': 'CORnet-S vision: trained unique',
    'cornet_untrained_plus_TEL': 'CORnet-S+TEL: trained-DNN unique',
    'cornet_untrained_plus_e5mistral': 'CORnet-S+e5: trained-DNN unique',
    'e5mistral_untrained': 'e5-mistral: trained unique',
}


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
    """Adjusted R2 used in the existing variance-partition notebook."""
    denom = n_obs - n_predictors - 1
    if denom <= 0:
        return np.full_like(r2, np.nan, dtype=float)
    return 1.0 - (((1.0 - r2) * (n_obs - 1)) / denom)


def nested_r2_for_half(pred, ctrl, bio_half):
    """Return raw and adjusted delta R2 arrays, shape (channels, time).

    This vectorizes independent nested regressions over channel x time.
    Each channel/timepoint has its own one-dimensional control and
    predictor vectors across images.
    """
    pred = pred[:, :63, :].reshape(pred.shape[0], -1).astype(float)
    ctrl = ctrl[:, :63, :].reshape(ctrl.shape[0], -1).astype(float)
    y = bio_half[:, :63, :].reshape(bio_half.shape[0], -1).astype(float)

    # Centering handles the intercept term.
    pred = pred - pred.mean(axis=0, keepdims=True)
    ctrl = ctrl - ctrl.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)

    syy = np.sum(y * y, axis=0)
    scc = np.sum(ctrl * ctrl, axis=0)
    spp = np.sum(pred * pred, axis=0)
    syc = np.sum(y * ctrl, axis=0)
    syp = np.sum(y * pred, axis=0)
    scp = np.sum(ctrl * pred, axis=0)

    eps = np.finfo(float).eps
    valid_y = syy > eps

    # Reduced model: y ~ control. Existing notebook floors negative R2 at 0.
    r2_reduced = np.zeros_like(syy)
    valid_red = valid_y & (scc > eps)
    r2_reduced[valid_red] = (syc[valid_red] ** 2) / (
        syy[valid_red] * scc[valid_red]
    )

    # Full two-predictor model: y ~ control + predictor.
    det = scc * spp - scp * scp
    r2_full = r2_reduced.copy()
    valid_full = valid_y & (det > eps)
    numerator = (
        syc[valid_full] ** 2 * spp[valid_full]
        - 2.0 * syc[valid_full] * syp[valid_full] * scp[valid_full]
        + syp[valid_full] ** 2 * scc[valid_full]
    )
    r2_full[valid_full] = numerator / (syy[valid_full] * det[valid_full])

    r2_reduced = np.nan_to_num(np.clip(r2_reduced, 0.0, 1.0))
    r2_full = np.nan_to_num(np.clip(r2_full, 0.0, 1.0))

    delta = r2_full - r2_reduced
    delta_adj = adjusted_r2(r2_full, y.shape[0], 2) - adjusted_r2(
        r2_reduced, y.shape[0], 1
    )
    return (
        delta.reshape(63, -1),
        delta_adj.reshape(63, -1),
        r2_full.reshape(63, -1),
        r2_reduced.reshape(63, -1),
    )


def compute_subject_delta(project_dir, sub, predictor, control, iterations, seed):
    pred = load_synth(project_dir, sub, predictor)
    ctrl = load_synth(project_dir, sub, control)
    bio, times = load_bio(project_dir, sub)
    bio = bio[:, :, :63, :]

    rng = np.random.default_rng(seed + sub)
    deltas = []
    deltas_adj = []
    fulls = []
    reduceds = []
    n_reps = bio.shape[1]
    for _ in range(iterations):
        # Match the older code's split-half target: average the held-in half.
        idx = resample(
            np.arange(n_reps),
            replace=False,
            random_state=int(rng.integers(0, np.iinfo(np.int32).max)),
        )[: n_reps // 2]
        bio_half = np.mean(np.delete(bio, idx, axis=1), axis=1)
        delta, delta_adj, r2_full, r2_reduced = nested_r2_for_half(
            pred, ctrl, bio_half
        )
        deltas.append(delta)
        deltas_adj.append(delta_adj)
        fulls.append(r2_full)
        reduceds.append(r2_reduced)

    return {
        'delta_r2': np.mean(deltas, axis=0),
        'delta_r2_adj': np.mean(deltas_adj, axis=0),
        'r2_full': np.mean(fulls, axis=0),
        'r2_reduced': np.mean(reduceds, axis=0),
        'times': times,
    }


def channel_mean(arr):
    return arr.mean(axis=1)


def summarize_curve(analysis, pretty, predictor, control, label, mat, times):
    mean = mat.mean(axis=0)
    peak_idx = int(np.argmax(mean))

    def win_mean(lo, hi):
        mask = (times >= lo) & (times <= hi)
        if not np.any(mask):
            return float('nan')
        return float(mat[:, mask].mean())

    return {
        'analysis': analysis,
        'contrast': pretty,
        'predictor': predictor,
        'control': control,
        'label': label,
        'n_subjects': len(SUBJECTS),
        'mean_delta_r2_adj_all_time': f'{float(mat.mean()):.8f}',
        'peak_delta_r2_adj': f'{float(mean[peak_idx]):.8f}',
        'peak_time_s': f'{float(times[peak_idx]):.6f}',
        'sem_at_peak': f'{float(mat[:, peak_idx].std(ddof=1) / np.sqrt(mat.shape[0])):.8f}',
        'mean_delta_r2_adj_0_200ms': f'{win_mean(0.0, 0.2):.8f}',
        'mean_delta_r2_adj_200_600ms': f'{win_mean(0.2, 0.6):.8f}',
    }


def plot_curves(curves, title, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap('tab10')
    for i, (pretty, mat, times) in enumerate(curves):
        mean = mat.mean(axis=0)
        sem = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
        color = cmap(i)
        ax.plot(times, mean, color=color, label=pretty, lw=1.6)
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
    parser = argparse.ArgumentParser(
        description='Compute nested-model delta R2 for variance partitioning.'
    )
    parser.add_argument(
        '--project_dir', default=PROJECT_DIR,
        help='Root directory holding linear_results and eeg_dataset.',
    )
    parser.add_argument(
        '--iterations', type=int, default=100,
        help='Number of split-half resampling iterations per subject.',
    )
    parser.add_argument(
        '--seed', type=int, default=20260510,
        help='Base random seed; per-subject seed is seed + subject index.',
    )
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    contrasts = []
    for predictor, control, label in TRAINED_VS_UNTRAINED:
        contrasts.append((
            'trained_vs_untrained',
            TRAINED_PRETTY.get(label, label),
            predictor,
            control,
            label,
        ))

    for layer in LAYERS:
        fusion = f'cornet_s_with_text_embedding_large_r2_v3_layerwise_{layer}'
        vision = f'cornet_s_r2_v3_layerwise_{layer}'
        contrasts.append((
            'layerwise_llm_unique',
            f'{layer}: LLM unique given DNN layer',
            fusion,
            vision,
            f'vision_layer_{layer}',
        ))
        contrasts.append((
            'layerwise_dnn_unique',
            f'{layer}: DNN-layer unique given TEL',
            fusion,
            'text_embedding_large_r2_v3',
            f'TEL_at_layer_{layer}',
        ))

    arrays = {}
    rows = []
    trained_curves = []
    layer_curves_llm = []
    layer_curves_dnn = []

    for analysis, pretty, predictor, control, label in contrasts:
        print(f'contrast: {analysis} | {pretty}', flush=True)
        subj_delta = []
        subj_delta_adj = []
        subj_full = []
        subj_reduced = []
        times = None
        for sub in SUBJECTS:
            out = compute_subject_delta(
                args.project_dir, sub, predictor, control,
                args.iterations, args.seed,
            )
            subj_delta.append(out['delta_r2'])
            subj_delta_adj.append(out['delta_r2_adj'])
            subj_full.append(out['r2_full'])
            subj_reduced.append(out['r2_reduced'])
            if times is None:
                times = out['times']

        delta = np.stack(subj_delta)
        delta_adj = np.stack(subj_delta_adj)
        full = np.stack(subj_full)
        reduced = np.stack(subj_reduced)
        key = f'{analysis}__{label}'
        arrays[f'{key}__delta_r2'] = delta
        arrays[f'{key}__delta_r2_adj'] = delta_adj
        arrays[f'{key}__r2_full'] = full
        arrays[f'{key}__r2_reduced'] = reduced
        arrays[f'{key}__times'] = times

        curve = channel_mean(delta_adj)
        rows.append(summarize_curve(
            analysis, pretty, predictor, control, label, curve, times
        ))
        if analysis == 'trained_vs_untrained':
            trained_curves.append((pretty, curve, times))
        elif analysis == 'layerwise_llm_unique':
            layer_curves_llm.append((pretty, curve, times))
        elif analysis == 'layerwise_dnn_unique':
            layer_curves_dnn.append((pretty, curve, times))

    summary_path = op.join(ANALYSIS_DIR, 'delta_r2_rebuttal_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    arrays_path = op.join(ANALYSIS_DIR, 'delta_r2_rebuttal_arrays.npz')
    np.savez_compressed(arrays_path, **arrays)

    plot_curves(
        trained_curves,
        'Nested-model delta R2: trained components controlling untrained variants',
        'Adjusted delta R2 (full - reduced, all-channel mean)',
        op.join(OUT_DIR, 'delta_r2_trained_vs_untrained.png'),
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, curves, title in (
        (axes[0], layer_curves_llm, 'Layer-wise LLM unique delta R2'),
        (axes[1], layer_curves_dnn, 'Layer-wise DNN-layer unique delta R2'),
    ):
        cmap = plt.get_cmap('viridis')
        for i, (pretty, mat, times) in enumerate(curves):
            mean = mat.mean(axis=0)
            sem = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
            color = cmap(i / max(1, len(curves) - 1))
            label = pretty.split(':', 1)[0]
            ax.plot(times, mean, color=color, label=label, lw=1.6)
            ax.fill_between(times, mean - sem, mean + sem,
                            color=color, alpha=0.18, lw=0)
        ax.axvline(0, color='k', lw=0.6, ls='--', alpha=0.6)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel('Time (s)')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(times[0], times[-1])
    axes[0].set_ylabel('Adjusted delta R2 (full - reduced, all-channel mean)')
    fig.tight_layout()
    out_layer = op.join(OUT_DIR, 'delta_r2_layerwise.png')
    fig.savefig(out_layer, dpi=200)
    plt.close(fig)

    print(f'wrote {summary_path}')
    print(f'wrote {arrays_path}')
    print(op.join(OUT_DIR, 'delta_r2_trained_vs_untrained.png'))
    print(out_layer)


if __name__ == '__main__':
    main()
