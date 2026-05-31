"""Cross-layer SHARED variance for the manuscript pc1000 baseline.

Computes the 3-way Borcard-Legendre partition for the headline manuscript
analysis (full CORnet-S, PCA-1000, paired with OpenAI text-embedding-large
of GPT-4 captions):

  DNN-only:  cornet_s_1000_features (legacy, r2_etc_new_eeg_v3_1k_combined)
  LLM-only:  text_embedding_large (rebuttal r2_v3 -- PROXY, see caveat)
  Fusion:    cornet_s_with_gpt4_features_embedded_5v_large_avg_pca_1000_features
             (legacy, r2_etc_new_eeg_v3_1k_combined)

Caveat: the LLM-only correlation file was never produced under the legacy
pipeline.  We substitute the rebuttal text_embedding_large_r2_v3 file
because it is the same OpenAI embedding model on the same GPT-4 captions.
Pipeline-mismatch bias is expected to be small but non-zero.

Source of R^2: the per-subject correlation file stores the iteration-
averaged Pearson r per (channel, time).  For a single-predictor regression
with centered data, R^2(y ~ pred) equals r^2.  We therefore use r^2 as our
R^2 estimate.  Granularity = per-subject channel x time (no per-iteration
arrays in these files).  SEM is computed across the n=10 subjects only.

Partition (per channel x time):
  SV         = R^2(DNN) + R^2(LLM) - R^2(fusion)
  UV_std(LLM) = R^2(fusion) - R^2(DNN)
  UV_std(DNN) = R^2(fusion) - R^2(LLM)
  conservation: R^2(fusion) = UV_std(LLM) + UV_std(DNN) + SV   (exact)

Outputs:
  - analysis/shared_variance_crosslayer_pc1000_summary.csv
  - analysis/shared_variance_crosslayer_pc1000_arrays.npz
  - rebuttal/figures/delta_r2_crosslayer_pc1000_shared.png
  - rebuttal/figures/delta_r2_crosslayer_pc1000_3way.png
"""

import csv
import os
import os.path as op

import matplotlib

# --- path configuration (portable defaults for public release) ---
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_DATA_ROOT = _os.environ.get('EEG_FUSION_DATA', _os.path.join(_REPO, 'data'))
# --- end path configuration ---

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR  = _DATA_ROOT
OUT_DIR      = _os.path.join(_REPO, 'rebuttal/figures')
ANALYSIS_DIR = _os.path.join(_REPO, 'analysis')
SUBJECTS     = list(range(1, 11))

# Correlation-file basenames per role.  All three files live under
# linear_results/sub-XX/correlation/correlation_<basename>.npy.
DNN_PC1000_BASENAME    = 'cornet_s_1000_features_all_r2_etc_new_eeg_v3_1k_combined'
LLM_PROXY_BASENAME     = 'text_embedding_large_r2_v3_all'
FUSION_PC1000_BASENAME = ('cornet_s_with_gpt4_features_embedded_5v_large_avg'
                         '_pca_1000_features_all_r2_etc_new_eeg_v3_1k_combined')


def load_corr(sub, basename):
    path = op.join(PROJECT_DIR, 'linear_results', f'sub-{sub:02d}',
                   'correlation', f'correlation_{basename}.npy')
    d = np.load(path, allow_pickle=True).item()
    return d['correlation'], np.asarray(d['times']), d.get('ch_names')


def winmean(mat, times, lo, hi):
    mask = (times >= lo) & (times <= hi)
    if not np.any(mask):
        return float('nan')
    return float(mat[:, mask].mean())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    r2_dnn_list, r2_llm_list, r2_fusion_list = [], [], []
    times = None
    for sub in SUBJECTS:
        r_dnn,    t_dnn,    _ = load_corr(sub, DNN_PC1000_BASENAME)
        r_llm,    t_llm,    _ = load_corr(sub, LLM_PROXY_BASENAME)
        r_fusion, t_fusion, _ = load_corr(sub, FUSION_PC1000_BASENAME)

        # Sanity: same time vector across files.
        if times is None:
            times = t_dnn
        assert np.allclose(t_dnn, t_llm) and np.allclose(t_dnn, t_fusion), \
            f'time vectors do not match for sub-{sub:02d}'

        # R^2 = r^2 for single-predictor regression with centering.
        # Negative correlations are folded to zero predictive R^2 by squaring.
        r2_dnn_list.append(r_dnn ** 2)
        r2_llm_list.append(r_llm ** 2)
        r2_fusion_list.append(r_fusion ** 2)

    r2_dnn    = np.stack(r2_dnn_list)     # (10, 63, 180)
    r2_llm    = np.stack(r2_llm_list)
    r2_fusion = np.stack(r2_fusion_list)

    sv          = r2_dnn + r2_llm - r2_fusion          # (10, 63, 180)
    uv_llm_std  = r2_fusion - r2_dnn
    uv_dnn_std  = r2_fusion - r2_llm

    # Channel-mean curves per subject for plotting/stats.
    sv_curve     = sv.mean(axis=1)         # (subj, time)
    uv_llm_curve = uv_llm_std.mean(axis=1)
    uv_dnn_curve = uv_dnn_std.mean(axis=1)

    # Save arrays.
    np.savez_compressed(
        op.join(ANALYSIS_DIR, 'shared_variance_crosslayer_pc1000_arrays.npz'),
        r2_dnn=r2_dnn, r2_llm=r2_llm, r2_fusion=r2_fusion,
        sv=sv, uv_llm_std=uv_llm_std, uv_dnn_std=uv_dnn_std,
        times=times,
    )

    # Summary row.
    mean_sv = sv_curve.mean(axis=0)
    pk_idx = int(np.argmax(mean_sv))
    summary = {
        'label': 'CORnet-S(pc1000) + gpt4-large-avg (manuscript headline)',
        'dnn_only_file':  f'correlation_{DNN_PC1000_BASENAME}.npy',
        'llm_only_file':  f'correlation_{LLM_PROXY_BASENAME}.npy  (PROXY: rebuttal r2_v3)',
        'fusion_file':    f'correlation_{FUSION_PC1000_BASENAME}.npy',
        'n_subjects':     len(SUBJECTS),
        'mean_sv_all_time':      f'{float(sv_curve.mean()):.8f}',
        'peak_sv':               f'{float(mean_sv[pk_idx]):.8f}',
        'peak_time_s':           f'{float(times[pk_idx]):.6f}',
        'sem_sv_at_peak':        f'{float(sv_curve[:, pk_idx].std(ddof=1) / np.sqrt(sv_curve.shape[0])):.8f}',
        'mean_sv_0_200ms':       f'{winmean(sv_curve, times, 0.0, 0.2):.8f}',
        'mean_sv_200_600ms':     f'{winmean(sv_curve, times, 0.2, 0.6):.8f}',
        'r2_dnn_peak':           f'{float(r2_dnn.mean(axis=1).mean(axis=0).max()):.8f}',
        'r2_llm_peak':           f'{float(r2_llm.mean(axis=1).mean(axis=0).max()):.8f}',
        'r2_fusion_peak':        f'{float(r2_fusion.mean(axis=1).mean(axis=0).max()):.8f}',
        'uv_llm_std_peak':       f'{float(uv_llm_std.mean(axis=1).mean(axis=0).max()):.8f}',
        'uv_dnn_std_peak':       f'{float(uv_dnn_std.mean(axis=1).mean(axis=0).max()):.8f}',
        'conservation_max_abs_err':
            f'{float(np.max(np.abs((sv + uv_llm_std + uv_dnn_std) - r2_fusion))):.2e}',
    }
    csv_path = op.join(ANALYSIS_DIR, 'shared_variance_crosslayer_pc1000_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    # SV-only line plot.
    fig, ax = plt.subplots(figsize=(8, 5))
    mean = sv_curve.mean(axis=0)
    sem  = sv_curve.std(axis=0, ddof=1) / np.sqrt(sv_curve.shape[0])
    ax.plot(times, mean, color='tab:purple', lw=1.8,
            label='CORnet-S pc1000 ∩ TEL (manuscript)')
    ax.fill_between(times, mean - sem, mean + sem,
                    color='tab:purple', alpha=0.22, lw=0)
    ax.axvline(0, color='k', lw=0.6, ls='--', alpha=0.6)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Shared R² = R²(DNN) + R²(LLM) − R²(fusion)')
    ax.set_title('Cross-layer shared variance: CORnet-S pc1000 + GPT-4-large-avg (manuscript headline)')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(times[0], times[-1])
    fig.tight_layout()
    fig.savefig(op.join(OUT_DIR, 'delta_r2_crosslayer_pc1000_shared.png'), dpi=200)
    plt.close(fig)

    # 3-panel partition figure.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, mat, title in (
        (axes[0], uv_llm_curve, 'UV(LLM) standard = R²(fusion) − R²(DNN)'),
        (axes[1], uv_dnn_curve, 'UV(DNN) standard = R²(fusion) − R²(LLM)'),
        (axes[2], sv_curve,     'Shared variance (DNN ∩ LLM)'),
    ):
        m = mat.mean(axis=0)
        s = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
        ax.plot(times, m, color='tab:purple', lw=1.8)
        ax.fill_between(times, m - s, m + s, color='tab:purple', alpha=0.22, lw=0)
        ax.axvline(0, color='k', lw=0.6, ls='--', alpha=0.6)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel('Time (s)')
        ax.set_title(title, fontsize=10)
        ax.set_xlim(times[0], times[-1])
    axes[0].set_ylabel('R² (channel mean, n=10 subjects)')
    fig.suptitle('Cross-layer (CORnet-S pc1000 + GPT-4-large-avg) 3-way partition',
                 y=1.02)
    fig.tight_layout()
    fig.savefig(op.join(OUT_DIR, 'delta_r2_crosslayer_pc1000_3way.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Console summary.
    print('=== Cross-layer pc1000 SV summary ===')
    for k, v in summary.items():
        print(f'  {k}: {v}')
    print()
    print('wrote:')
    print(' ', csv_path)
    print(' ', op.join(ANALYSIS_DIR, 'shared_variance_crosslayer_pc1000_arrays.npz'))
    print(' ', op.join(OUT_DIR, 'delta_r2_crosslayer_pc1000_shared.png'))
    print(' ', op.join(OUT_DIR, 'delta_r2_crosslayer_pc1000_3way.png'))


if __name__ == '__main__':
    main()
