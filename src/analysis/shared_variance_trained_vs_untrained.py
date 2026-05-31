"""Shared variance between trained LLM and trained / untrained DCNN.

Tests whether the DNN-LLM overlap is *learned* or *architectural* by
comparing the Borcard-Legendre 3-way partition SV(DNN, LLM) for
trained-CORnet-S vs untrained-CORnet-S, with the LLM (trained TEL) held
fixed in both cases.

All five inputs are from the rebuttal r2_v3 pipeline (internally
consistent; pc1000 cannot be used here because the untrained-CORnet
encoding was never produced under the legacy pipeline):

  trained DNN-only       cornet_s_r2_v3_pc500
  untrained DNN-only     cornet_s_untrained_r2_v3
  LLM-only (trained)     text_embedding_large_r2_v3
  trained fusion         cornet_s_with_text_embedding_large_r2_v3_pc500
  untrained fusion       cornet_s_untrained_with_text_embedding_large_r2_v3

Per channel x time per subject:

  SV(trained-DNN,   LLM) = R^2(tr-DNN)  + R^2(LLM) - R^2(tr-fusion)
  SV(untrained-DNN, LLM) = R^2(unt-DNN) + R^2(LLM) - R^2(unt-fusion)

  UV(LLM | tr-DNN)   = R^2(tr-fusion)  - R^2(tr-DNN)
  UV(LLM | unt-DNN)  = R^2(unt-fusion) - R^2(unt-DNN)
  UV(tr-DNN | LLM)   = R^2(tr-fusion)  - R^2(LLM)
  UV(unt-DNN | LLM)  = R^2(unt-fusion) - R^2(LLM)

Interpretation:
  SV(unt-DNN, LLM) ≈ SV(tr-DNN, LLM)   → overlap is architectural / image-
                                          statistic-driven, not learned.
  SV(tr-DNN, LLM)  >> SV(unt-DNN, LLM) → training carves new shared axes
                                          with the LLM's semantic content.

R^2 source: per-subject correlation files store the iteration-averaged
Pearson r per (channel, time); R^2 = r^2 for a single-predictor regression
on centered data. SEM is across the n=10 subjects only.

Outputs:
  - analysis/sv_dnn_llm_trained_vs_untrained_summary.csv
  - analysis/sv_dnn_llm_trained_vs_untrained_arrays.npz
  - rebuttal/figures/sv_dnn_llm_trained_vs_untrained.png             (SV only)
  - rebuttal/figures/sv_dnn_llm_trained_vs_untrained_with_uv.png     (UV(LLM) + UV(DCNN) + SV)
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

PREFIXES = {
    'dnn_trained':     'cornet_s_r2_v3_pc500_all',
    'dnn_untrained':   'cornet_s_untrained_r2_v3_all',
    'llm':             'text_embedding_large_r2_v3_all',
    'fus_trained':     'cornet_s_with_text_embedding_large_r2_v3_pc500_all',
    'fus_untrained':   'cornet_s_untrained_with_text_embedding_large_r2_v3_all',
}


def load_r2(sub, basename):
    """Return R^2 = r^2 per (channel, time) for one subject and its time axis."""
    path = op.join(PROJECT_DIR, 'linear_results', f'sub-{sub:02d}',
                   'correlation', f'correlation_{basename}.npy')
    d = np.load(path, allow_pickle=True).item()
    return d['correlation'] ** 2, np.asarray(d['times'])


def winmean(mat, times, lo, hi):
    mask = (times >= lo) & (times <= hi)
    if not np.any(mask):
        return float('nan')
    return float(mat[:, mask].mean())


def t_one_samp(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return float(x.mean()), float('nan'), float('nan'), n
    m = x.mean()
    s = x.std(ddof=1) / np.sqrt(n)
    return float(m), float(s), float(m / s) if s > 0 else float('nan'), n


def sig_label(t):
    a = abs(t)
    if a > 4.781: return 'p<0.001'
    if a > 3.250: return 'p<0.01'
    if a > 2.262: return 'p<0.05'
    if a > 1.833: return 'p<0.10'
    return 'n.s.'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    bag = {k: [] for k in PREFIXES}
    times = None
    for sub in SUBJECTS:
        for role, base in PREFIXES.items():
            r2, t = load_r2(sub, base)
            bag[role].append(r2)
            if times is None:
                times = t
            else:
                assert np.allclose(times, t), f'time-vector mismatch on sub-{sub:02d} {role}'
    for k in bag:
        bag[k] = np.stack(bag[k])           # (10, 63, 180)

    r2_dnn_tr  = bag['dnn_trained']
    r2_dnn_un  = bag['dnn_untrained']
    r2_llm     = bag['llm']
    r2_fus_tr  = bag['fus_trained']
    r2_fus_un  = bag['fus_untrained']

    sv_tr  = r2_dnn_tr  + r2_llm - r2_fus_tr      # SV(trained-DNN,   LLM)
    sv_un  = r2_dnn_un  + r2_llm - r2_fus_un      # SV(untrained-DNN, LLM)
    uv_llm_tr = r2_fus_tr  - r2_dnn_tr             # UV(LLM | trained-DNN)
    uv_llm_un = r2_fus_un  - r2_dnn_un             # UV(LLM | untrained-DNN)
    uv_dnn_tr = r2_fus_tr  - r2_llm                # UV(trained-DNN | LLM)
    uv_dnn_un = r2_fus_un  - r2_llm                # UV(untrained-DNN | LLM)
    delta_sv = sv_tr - sv_un                       # paired: training-induced extra SV
    delta_uv_dnn = uv_dnn_tr - uv_dnn_un           # paired: trained-DNN unique gain

    # Channel means per subject for stats / plots.
    cm = lambda a: a.mean(axis=1)                  # (subj, time)
    sv_tr_c       = cm(sv_tr)
    sv_un_c       = cm(sv_un)
    uv_llm_tr_c   = cm(uv_llm_tr)
    uv_llm_un_c   = cm(uv_llm_un)
    uv_dnn_tr_c   = cm(uv_dnn_tr)
    uv_dnn_un_c   = cm(uv_dnn_un)
    delta_c       = cm(delta_sv)
    delta_uv_dnn_c = cm(delta_uv_dnn)

    # Save arrays.
    np.savez_compressed(
        op.join(ANALYSIS_DIR, 'sv_dnn_llm_trained_vs_untrained_arrays.npz'),
        r2_dnn_trained=r2_dnn_tr, r2_dnn_untrained=r2_dnn_un, r2_llm=r2_llm,
        r2_fusion_trained=r2_fus_tr, r2_fusion_untrained=r2_fus_un,
        sv_trained=sv_tr, sv_untrained=sv_un,
        uv_llm_given_trained=uv_llm_tr, uv_llm_given_untrained=uv_llm_un,
        uv_dnn_trained_given_llm=uv_dnn_tr,
        uv_dnn_untrained_given_llm=uv_dnn_un,
        delta_sv_trained_minus_untrained=delta_sv,
        delta_uv_dnn_trained_minus_untrained=delta_uv_dnn,
        times=times,
    )

    # Summary rows: each contrast at peak + windowed means.
    rows = []
    for label, curve in (
        ('SV(trained-DNN ∩ LLM)',   sv_tr_c),
        ('SV(untrained-DNN ∩ LLM)', sv_un_c),
        ('UV(LLM | trained-DNN)',   uv_llm_tr_c),
        ('UV(LLM | untrained-DNN)', uv_llm_un_c),
        ('UV(trained-DNN | LLM)',   uv_dnn_tr_c),
        ('UV(untrained-DNN | LLM)', uv_dnn_un_c),
        ('UV(DNN | LLM): trained − untrained (paired)', delta_uv_dnn_c),
        ('SV_trained − SV_untrained (paired)', delta_c),
    ):
        gmean = curve.mean(axis=0)
        pk_idx = int(np.argmax(gmean))
        m_pk, sem_pk, t_pk, _ = t_one_samp(curve[:, pk_idx])
        m_e,  sem_e, t_e, _ = t_one_samp(curve[:, (times>=0.05)&(times<=0.2)].mean(axis=1))
        m_l,  sem_l, t_l, _ = t_one_samp(curve[:, (times>=0.2)&(times<=0.6)].mean(axis=1))
        rows.append({
            'contrast': label,
            'peak_value':         f'{m_pk:+.6f}',
            'peak_sem':           f'{sem_pk:.6f}',
            'peak_time_s':        f'{float(times[pk_idx]):.4f}',
            'peak_t9':            f'{t_pk:+.2f}',
            'peak_sig':           sig_label(t_pk),
            'mean_50_200ms':      f'{m_e:+.6f}',
            'sem_50_200ms':       f'{sem_e:.6f}',
            't9_50_200ms':        f'{t_e:+.2f}',
            'sig_50_200ms':       sig_label(t_e),
            'mean_200_600ms':     f'{m_l:+.6f}',
            'sem_200_600ms':      f'{sem_l:.6f}',
            't9_200_600ms':       f'{t_l:+.2f}',
            'sig_200_600ms':      sig_label(t_l),
        })
    csv_path = op.join(ANALYSIS_DIR, 'sv_dnn_llm_trained_vs_untrained_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Figure 1: SV trained vs untrained, single panel.
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, mat, color in (
        ('trained CORnet-S ∩ trained LLM',   sv_tr_c, 'tab:blue'),
        ('untrained CORnet-S ∩ trained LLM', sv_un_c, 'tab:red'),
    ):
        m = mat.mean(axis=0)
        s = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
        ax.plot(times, m, color=color, lw=1.8, label=label)
        ax.fill_between(times, m - s, m + s, color=color, alpha=0.20, lw=0)
    # paired difference as light black line
    md = delta_c.mean(axis=0)
    sd = delta_c.std(axis=0, ddof=1) / np.sqrt(delta_c.shape[0])
    ax.plot(times, md, color='k', lw=1.2, ls='--',
            label='paired Δ (trained − untrained)')
    ax.fill_between(times, md - sd, md + sd, color='k', alpha=0.10, lw=0)
    ax.axvline(0, color='k', lw=0.6, ls=':', alpha=0.6)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Shared R²(DCNN ∩ LLM)  (channel mean, n=10)')
    ax.set_title('Shared variance: trained LLM ∩ trained vs untrained CORnet-S')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(times[0], times[-1])
    fig.tight_layout()
    fig.savefig(op.join(OUT_DIR, 'sv_dnn_llm_trained_vs_untrained.png'), dpi=200)
    plt.close(fig)

    # Figure 2: full trained-vs-untrained 3-way partition.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, curves, title in (
        (axes[0],
         [('UV(LLM | trained-DNN)',   uv_llm_tr_c, 'tab:blue'),
          ('UV(LLM | untrained-DNN)', uv_llm_un_c, 'tab:red')],
         'LLM unique variance, given trained vs untrained DCNN'),
        (axes[1],
         [('UV(trained-DNN | LLM)',   uv_dnn_tr_c, 'tab:blue'),
          ('UV(untrained-DNN | LLM)', uv_dnn_un_c, 'tab:red')],
         'DCNN unique variance, given LLM'),
        (axes[2],
         [('SV(trained-DNN ∩ LLM)',   sv_tr_c, 'tab:blue'),
          ('SV(untrained-DNN ∩ LLM)', sv_un_c, 'tab:red')],
         'Shared variance (DCNN ∩ LLM): trained vs untrained DCNN'),
    ):
        for label, mat, color in curves:
            m = mat.mean(axis=0)
            s = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
            ax.plot(times, m, color=color, lw=1.8, label=label)
            ax.fill_between(times, m - s, m + s, color=color, alpha=0.20, lw=0)
        ax.axvline(0, color='k', lw=0.6, ls=':', alpha=0.6)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xlabel('Time (s)')
        ax.set_title(title, fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlim(times[0], times[-1])
    axes[0].set_ylabel('R² (channel mean)')
    fig.tight_layout()
    fig.savefig(op.join(OUT_DIR, 'sv_dnn_llm_trained_vs_untrained_with_uv.png'),
                dpi=200)
    plt.close(fig)

    # Console summary.
    print('=== SV(DCNN ∩ LLM): trained vs untrained DCNN ===')
    for row in rows:
        print(f"  {row['contrast']:40s}  peak {row['peak_value']:>10s}  "
              f"@ {row['peak_time_s']}s  t(9)={row['peak_t9']:>6s}  {row['peak_sig']}")
        print(f"    {'  50-200ms':12s}  mean {row['mean_50_200ms']:>10s}  "
              f"t(9)={row['t9_50_200ms']:>6s}  {row['sig_50_200ms']}")
        print(f"    {'  200-600ms':12s}  mean {row['mean_200_600ms']:>10s}  "
              f"t(9)={row['t9_200_600ms']:>6s}  {row['sig_200_600ms']}")
    print()
    print('wrote:')
    print(' ', csv_path)
    print(' ', op.join(ANALYSIS_DIR, 'sv_dnn_llm_trained_vs_untrained_arrays.npz'))
    print(' ', op.join(OUT_DIR, 'sv_dnn_llm_trained_vs_untrained.png'))
    print(' ', op.join(OUT_DIR, 'sv_dnn_llm_trained_vs_untrained_with_uv.png'))


if __name__ == '__main__':
    main()
