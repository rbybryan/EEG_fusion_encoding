"""Conditional trained-CORnet / TEL variance partition for the rebuttal.

Question:
    How much of the trained CORnet-S / TEL overlap remains after controlling
    the architectural/image-statistic baseline captured by untrained CORnet-S?

Per subject, channel, timepoint:

    U = untrained CORnet-S prediction
    D = trained CORnet-S prediction
    L = TEL prediction

    UV(D | U)     = adjR2(U + D)     - adjR2(U)
    UV(D | U + L) = adjR2(U + L + D) - adjR2(U + L)
    UV(D | L)     = adjR2(L + D)     - adjR2(L)
    UV(L | D)     = adjR2(D + L)     - adjR2(D)
    SV(D ∩ L | U) = UV(D | U) - UV(D | U + L)
                 = adjR2(U + D) + adjR2(U + L)
                   - adjR2(U + D + L) - adjR2(U)

The biological target uses the same split-half resampling convention as the
other rebuttal variance analyses.

For the trained CORnet-S / TEL terms, this script uses the legacy cleaned
pc1000 triplet from <data>/linear_results_combined. The
untrained CORnet-S term remains the rebuttal untrained control from
<data>/linear_results.

Outputs:
  - analysis/conditional_sv_trained_cornet_tel_summary.csv
  - analysis/conditional_sv_trained_cornet_tel_arrays.npz
  - rebuttal/figures/conditional_sv_trained_cornet_tel.png
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

PROJECT_DIR = _DATA_ROOT
LEGACY_SYNTH_DIR = _os.path.join(_DATA_ROOT, 'linear_results_combined')
OUT_DIR = _os.path.join(_REPO, 'rebuttal/figures')
ANALYSIS_DIR = _os.path.join(_REPO, 'analysis')
SUBJECTS = list(range(1, 11))
ITERATIONS = 100
SEED = 20260510

TRAINED_CORNET = 'cornet_s_1000_features_all_r2_etc_new_eeg_v3_1k_combined'
UNTRAINED_CORNET = 'cornet_s_untrained_r2_v3'
TEL = ('gpt4_features_embedded_5v_large_cleaned_avg_pca_1000_features_all'
       '_ridge_r2_etc_new_eeg_v3_1k_combined')
LEGACY_SYNTH_NAMES = {TRAINED_CORNET, TEL}
TINY = np.finfo(float).tiny
REL_TOL = 1e-12


def load_synth(sub, name):
    if name in LEGACY_SYNTH_NAMES:
        base = op.join(LEGACY_SYNTH_DIR, f'sub-{sub:02d}', 'synthetic_eeg_data')
    else:
        base = op.join(PROJECT_DIR, 'linear_results', f'sub-{sub:02d}',
                       'synthetic_eeg_data')
    for cand in (f'{name}.npy', f'{name}_all.npy'):
        path = op.join(base, cand)
        if op.exists(path):
            return np.load(path, allow_pickle=True).item()['synthetic_data'][:, :63, :]
    raise FileNotFoundError(f'no synth file for {name} under {base}')


def load_bio(sub):
    path = op.join(PROJECT_DIR, 'eeg_dataset', 'preprocessed_eeg_data_v1',
                   f'eeg_sub-{sub:02d}_split-test.npy')
    d = np.load(path, allow_pickle=True).item()
    return d['preprocessed_eeg_data'][:, :, :63, :], np.asarray(d['times'])


def split_half_means(bio, iterations, seed, sub):
    """Build all split-half target means without repeated large tensor copies."""
    n_reps = bio.shape[1]
    weights = np.zeros((iterations, n_reps), dtype=np.result_type(bio.dtype, np.float32))
    rng = np.random.default_rng(seed + sub)
    for i in range(iterations):
        seed_i = int(rng.integers(0, np.iinfo(np.int32).max))
        idx = np.random.RandomState(seed_i).permutation(n_reps)[:n_reps // 2]
        weights[i, :] = 1.0 / (n_reps - len(idx))
        weights[i, idx] = 0.0
    return np.tensordot(weights, bio, axes=(1, 1))


def center_flat(a):
    n = a.shape[0]
    x = a.reshape(n, -1).astype(float)
    return x - x.mean(axis=0, keepdims=True)


def adjusted_r2(r2, n_obs, n_predictors):
    denom = n_obs - n_predictors - 1
    if denom <= 0:
        return np.full_like(r2, np.nan, dtype=float)
    return 1.0 - (((1.0 - r2) * (n_obs - 1)) / denom)


def safe_div(num, den):
    out = np.zeros_like(num)
    ok = den > TINY
    out[ok] = num[ok] / den[ok]
    return out


def residualize_one(x, u, suu):
    beta = safe_div(np.sum(u * x, axis=0), suu)
    return x - u * beta[None, :]


def residualize_two(x, u, l, suu, sll, sul, det):
    sux = np.sum(u * x, axis=0)
    slx = np.sum(l * x, axis=0)
    bu = np.zeros_like(det)
    bl = np.zeros_like(det)
    scale = np.maximum(suu * sll, TINY)
    ok = (suu > TINY) & (sll > TINY) & (det > REL_TOL * scale)
    bu[ok] = ((sux[ok] * sll[ok]) - (slx[ok] * sul[ok])) / det[ok]
    bl[ok] = ((slx[ok] * suu[ok]) - (sux[ok] * sul[ok])) / det[ok]
    return x - u * bu[None, :] - l * bl[None, :]


def compute_subject(sub):
    d = center_flat(load_synth(sub, TRAINED_CORNET))
    u = center_flat(load_synth(sub, UNTRAINED_CORNET))
    l = center_flat(load_synth(sub, TEL))
    bio, times = load_bio(sub)

    suu = np.sum(u * u, axis=0)
    sll = np.sum(l * l, axis=0)
    sul = np.sum(u * l, axis=0)
    det_ul = (suu * sll) - (sul * sul)

    d_res_u = residualize_one(d, u, suu)
    sdd_u = np.sum(d_res_u * d_res_u, axis=0)
    d_res_ul = residualize_two(d, u, l, suu, sll, sul, det_ul)
    sdd_ul = np.sum(d_res_ul * d_res_ul, axis=0)
    d_res_l = residualize_one(d, l, sll)
    sdd_l = np.sum(d_res_l * d_res_l, axis=0)
    sdd = np.sum(d * d, axis=0)
    l_res_d = residualize_one(l, d, sdd)
    sll_d = np.sum(l_res_d * l_res_d, axis=0)

    uv_d_u_sum = np.zeros((63, bio.shape[-1]), dtype=float)
    uv_d_ul_sum = np.zeros_like(uv_d_u_sum)
    uv_d_l_sum = np.zeros_like(uv_d_u_sum)
    uv_l_d_sum = np.zeros_like(uv_d_u_sum)
    sv_dl_u_sum = np.zeros_like(uv_d_u_sum)

    for bio_half in split_half_means(bio, ITERATIONS, SEED, sub):
        y = center_flat(bio_half)
        n_obs = y.shape[0]
        syy = np.sum(y * y, axis=0)

        y_res_u = residualize_one(y, u, suu)
        sse_u = np.sum(y_res_u * y_res_u, axis=0)
        r2_u = np.zeros_like(syy)
        ok = syy > TINY
        r2_u[ok] = 1.0 - (sse_u[ok] / syy[ok])
        r2_u = np.nan_to_num(np.clip(r2_u, 0.0, 1.0))

        y_res_ul = residualize_two(y, u, l, suu, sll, sul, det_ul)
        sse_ul = np.sum(y_res_ul * y_res_ul, axis=0)
        r2_ul = np.zeros_like(syy)
        r2_ul[ok] = 1.0 - (sse_ul[ok] / syy[ok])
        r2_ul = np.nan_to_num(np.clip(r2_ul, 0.0, 1.0))

        y_res_l = residualize_one(y, l, sll)
        sse_l = np.sum(y_res_l * y_res_l, axis=0)
        r2_l = np.zeros_like(syy)
        r2_l[ok] = 1.0 - (sse_l[ok] / syy[ok])
        r2_l = np.nan_to_num(np.clip(r2_l, 0.0, 1.0))

        y_res_d = residualize_one(y, d, sdd)
        sse_d = np.sum(y_res_d * y_res_d, axis=0)
        r2_d = np.zeros_like(syy)
        r2_d[ok] = 1.0 - (sse_d[ok] / syy[ok])
        r2_d = np.nan_to_num(np.clip(r2_d, 0.0, 1.0))

        syd_u = np.sum(y_res_u * d_res_u, axis=0)
        delta_d_u = np.zeros_like(syy)
        ok_d_u = (syy > TINY) & (sdd_u > TINY)
        delta_d_u[ok_d_u] = (syd_u[ok_d_u] ** 2) / (sdd_u[ok_d_u] * syy[ok_d_u])
        r2_ud = np.nan_to_num(np.clip(r2_u + delta_d_u, 0.0, 1.0))

        syd_ul = np.sum(y_res_ul * d_res_ul, axis=0)
        delta_d_ul = np.zeros_like(syy)
        ok_d_ul = (syy > TINY) & (sdd_ul > TINY)
        delta_d_ul[ok_d_ul] = (
            (syd_ul[ok_d_ul] ** 2) / (sdd_ul[ok_d_ul] * syy[ok_d_ul])
        )
        r2_uld = np.nan_to_num(np.clip(r2_ul + delta_d_ul, 0.0, 1.0))

        syd_l = np.sum(y_res_l * d_res_l, axis=0)
        delta_d_l = np.zeros_like(syy)
        ok_d_l = (syy > TINY) & (sdd_l > TINY)
        delta_d_l[ok_d_l] = (syd_l[ok_d_l] ** 2) / (sdd_l[ok_d_l] * syy[ok_d_l])
        r2_ld = np.nan_to_num(np.clip(r2_l + delta_d_l, 0.0, 1.0))

        syl_d = np.sum(y_res_d * l_res_d, axis=0)
        delta_l_d = np.zeros_like(syy)
        ok_l_d = (syy > TINY) & (sll_d > TINY)
        delta_l_d[ok_l_d] = (syl_d[ok_l_d] ** 2) / (sll_d[ok_l_d] * syy[ok_l_d])
        r2_dl = np.nan_to_num(np.clip(r2_d + delta_l_d, 0.0, 1.0))

        uv1 = adjusted_r2(r2_ud, n_obs, 2) - adjusted_r2(r2_u, n_obs, 1)
        uv2 = adjusted_r2(r2_uld, n_obs, 3) - adjusted_r2(r2_ul, n_obs, 2)
        uv3 = adjusted_r2(r2_ld, n_obs, 2) - adjusted_r2(r2_l, n_obs, 1)
        uv4 = adjusted_r2(r2_dl, n_obs, 2) - adjusted_r2(r2_d, n_obs, 1)
        uv_d_u_sum += uv1.reshape(63, -1)
        uv_d_ul_sum += uv2.reshape(63, -1)
        uv_d_l_sum += uv3.reshape(63, -1)
        uv_l_d_sum += uv4.reshape(63, -1)
        sv_dl_u_sum += (uv1 - uv2).reshape(63, -1)

    return {
        'uv_trained_cornet_given_untrained_cornet': uv_d_u_sum / ITERATIONS,
        'uv_trained_cornet_given_untrained_cornet_tel': uv_d_ul_sum / ITERATIONS,
        'uv_trained_cornet_given_tel': uv_d_l_sum / ITERATIONS,
        'uv_tel_given_trained_cornet': uv_l_d_sum / ITERATIONS,
        'sv_trained_cornet_tel_given_untrained_cornet': sv_dl_u_sum / ITERATIONS,
        'times': times,
    }


def t_one_samp(x):
    x = np.asarray(x, float)
    m = x.mean()
    sem = x.std(ddof=1) / np.sqrt(len(x))
    return float(m), float(sem), float(m / sem) if sem > 0 else float('nan')


def sig_label(t):
    a = abs(t)
    if a > 4.781: return 'p<0.001'
    if a > 3.250: return 'p<0.01'
    if a > 2.262: return 'p<0.05'
    if a > 1.833: return 'p<0.10'
    return 'n.s.'


def summarize(label, curve, times):
    mean = curve.mean(axis=0)
    pk_idx = int(np.argmax(mean))
    m_pk, sem_pk, t_pk = t_one_samp(curve[:, pk_idx])
    m_e, sem_e, t_e = t_one_samp(curve[:, (times >= 0.05) & (times <= 0.2)].mean(axis=1))
    m_l, sem_l, t_l = t_one_samp(curve[:, (times >= 0.2) & (times <= 0.6)].mean(axis=1))
    return {
        'contrast': label,
        'peak_value': f'{m_pk:+.6f}',
        'peak_sem': f'{sem_pk:.6f}',
        'peak_time_s': f'{float(times[pk_idx]):.4f}',
        'peak_t9': f'{t_pk:+.2f}',
        'peak_sig': sig_label(t_pk),
        'mean_50_200ms': f'{m_e:+.6f}',
        'sem_50_200ms': f'{sem_e:.6f}',
        't9_50_200ms': f'{t_e:+.2f}',
        'sig_50_200ms': sig_label(t_e),
        'mean_200_600ms': f'{m_l:+.6f}',
        'sem_200_600ms': f'{sem_l:.6f}',
        't9_200_600ms': f'{t_l:+.2f}',
        'sig_200_600ms': sig_label(t_l),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    arrays = {
        'uv_trained_cornet_given_tel': [],
        'uv_tel_given_trained_cornet': [],
        'uv_trained_cornet_given_untrained_cornet': [],
        'uv_trained_cornet_given_untrained_cornet_tel': [],
        'sv_trained_cornet_tel_given_untrained_cornet': [],
    }
    times = None
    for sub in SUBJECTS:
        out = compute_subject(sub)
        for k in arrays:
            arrays[k].append(out[k])
        if times is None:
            times = out['times']
        print(f'sub-{sub:02d} done', flush=True)

    for k in arrays:
        arrays[k] = np.stack(arrays[k])
    arrays['times'] = times

    np.savez_compressed(
        op.join(ANALYSIS_DIR, 'conditional_sv_trained_cornet_tel_arrays.npz'),
        **arrays,
    )

    curves = {
        'UV(trained CORnet-S | TEL)':
            arrays['uv_trained_cornet_given_tel'].mean(axis=1),
        'UV(TEL | trained CORnet-S)':
            arrays['uv_tel_given_trained_cornet'].mean(axis=1),
        'UV(trained CORnet-S | untrained CORnet-S + TEL)':
            arrays['uv_trained_cornet_given_untrained_cornet_tel'].mean(axis=1),
        'SV(trained CORnet-S ∩ TEL | untrained CORnet-S)':
            arrays['sv_trained_cornet_tel_given_untrained_cornet'].mean(axis=1),
    }
    rows = [summarize(label, curve, times) for label, curve in curves.items()]
    for row in rows:
        row.update({
            'trained_cornet_file': f'{TRAINED_CORNET}.npy',
            'tel_file': f'{TEL}.npy',
            'untrained_cornet_file': f'{UNTRAINED_CORNET}.npy',
            'trained_tel_source_dir': LEGACY_SYNTH_DIR,
            'untrained_source_dir': op.join(PROJECT_DIR, 'linear_results'),
        })

    csv_path = op.join(ANALYSIS_DIR, 'conditional_sv_trained_cornet_tel_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    styles = {
        'UV(trained CORnet-S | TEL)': ('#1f77b4', '-'),
        'UV(TEL | trained CORnet-S)': ('#ff7f0e', '-'),
        'UV(trained CORnet-S | untrained CORnet-S + TEL)': ('#08519c', '-'),
        'SV(trained CORnet-S ∩ TEL | untrained CORnet-S)': ('#6a3d9a', '-'),
    }
    for label, curve in curves.items():
        color, linestyle = styles[label]
        mean = curve.mean(axis=0)
        sem = curve.std(axis=0, ddof=1) / np.sqrt(curve.shape[0])
        ax.plot(times, mean, color=color, lw=1.8, ls=linestyle, label=label)
        ax.fill_between(times, mean - sem, mean + sem,
                        color=color, alpha=0.18, lw=0)
    ax.axvline(0, color='k', lw=0.6, ls=':', alpha=0.6)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Adjusted ΔR² / shared R² (channel mean)')
    ax.set_title('Conditional partition: trained CORnet-S pc1000 and TEL pc1000 | untrained CORnet-S')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(times[0], times[-1])
    fig.tight_layout()
    fig.savefig(op.join(OUT_DIR, 'conditional_sv_trained_cornet_tel.png'), dpi=200)
    plt.close(fig)

    print('=== Conditional trained CORnet-S / TEL partition ===')
    for row in rows:
        print(f"  {row['contrast']}")
        print(f"    peak {row['peak_value']} @ {row['peak_time_s']}s "
              f"t(9)={row['peak_t9']} {row['peak_sig']}")
        print(f"    50-200ms mean {row['mean_50_200ms']} "
              f"t(9)={row['t9_50_200ms']} {row['sig_50_200ms']}")
        print(f"    200-600ms mean {row['mean_200_600ms']} "
              f"t(9)={row['t9_200_600ms']} {row['sig_200_600ms']}")
    print('wrote:')
    print(' ', csv_path)
    print(' ', op.join(ANALYSIS_DIR, 'conditional_sv_trained_cornet_tel_arrays.npz'))
    print(' ', op.join(OUT_DIR, 'conditional_sv_trained_cornet_tel.png'))


if __name__ == '__main__':
    main()
