"""Conditional trained-CORnet / e5-Mistral variance partition for rebuttal.

Notation:
    U  = untrained CORnet-S prediction
    D  = trained CORnet-S prediction
    L  = trained e5-Mistral prediction
    L0 = untrained e5-Mistral prediction

Reported curves:
    UV(D | L)
    UV(L | D)
    UV(D | U + L)
    UV(L | D + L0)
    SV(D ∩ L | U) = UV(D | U) - UV(D | U + L)

All terms use adjusted R2 and the same split-half target resampling as the
other rebuttal variance analyses.
"""

import csv
import os
import os.path as op
import argparse

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
OUT_DIR = _os.path.join(_REPO, 'rebuttal/figures')
ANALYSIS_DIR = _os.path.join(_REPO, 'analysis')
SUBJECT_CACHE_DIR = op.join(
    ANALYSIS_DIR, 'conditional_sv_trained_cornet_mistral_subjects'
)
SUBJECTS = list(range(1, 11))
ITERATIONS = 100
SEED = 20260510

TRAINED_CORNET = 'cornet_s_r2_v3_e5mistral'
UNTRAINED_CORNET = 'cornet_s_r2_v3_e5mistral_untrained'
MISTRAL = 'e5-mistral-7b-instruct_cleaned_r2_v3_e5mistral'
UNTRAINED_MISTRAL = 'e5-mistral-7b-instruct_untrained_cleaned_r2_v3_e5mistral_untrained'
TINY = np.finfo(float).tiny
REL_TOL = 1e-12
ARRAY_KEYS = (
    'uv_trained_cornet_given_mistral',
    'uv_mistral_given_trained_cornet',
    'uv_trained_cornet_given_untrained_cornet_mistral',
    'uv_mistral_given_trained_cornet_untrained_mistral',
    'sv_trained_cornet_mistral_given_untrained_cornet',
)


def load_synth(sub, name):
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


def residualize_one(x, c, scc):
    beta = safe_div(np.sum(c * x, axis=0), scc)
    return x - c * beta[None, :]


def residualize_two(x, c1, c2, s11, s22, s12, det):
    s1x = np.sum(c1 * x, axis=0)
    s2x = np.sum(c2 * x, axis=0)
    b1 = np.zeros_like(det)
    b2 = np.zeros_like(det)
    scale = np.maximum(s11 * s22, TINY)
    ok = (s11 > TINY) & (s22 > TINY) & (det > REL_TOL * scale)
    b1[ok] = ((s1x[ok] * s22[ok]) - (s2x[ok] * s12[ok])) / det[ok]
    b2[ok] = ((s2x[ok] * s11[ok]) - (s1x[ok] * s12[ok])) / det[ok]
    return x - c1 * b1[None, :] - c2 * b2[None, :]


def r2_from_residual(y, y_res):
    syy = np.sum(y * y, axis=0)
    sse = np.sum(y_res * y_res, axis=0)
    r2 = np.zeros_like(syy)
    ok = syy > TINY
    r2[ok] = 1.0 - (sse[ok] / syy[ok])
    return np.nan_to_num(np.clip(r2, 0.0, 1.0))


def delta_adj_from_residuals(y, y_res, pred_res, r2_reduced, n_obs, k_reduced):
    syy = np.sum(y * y, axis=0)
    spp = np.sum(pred_res * pred_res, axis=0)
    syp = np.sum(y_res * pred_res, axis=0)
    delta = np.zeros_like(syy)
    ok = (syy > TINY) & (spp > TINY)
    delta[ok] = (syp[ok] ** 2) / (spp[ok] * syy[ok])
    r2_full = np.nan_to_num(np.clip(r2_reduced + delta, 0.0, 1.0))
    return adjusted_r2(r2_full, n_obs, k_reduced + 1) - adjusted_r2(
        r2_reduced, n_obs, k_reduced
    )


def compute_subject(sub):
    d = center_flat(load_synth(sub, TRAINED_CORNET))
    u = center_flat(load_synth(sub, UNTRAINED_CORNET))
    l = center_flat(load_synth(sub, MISTRAL))
    l0 = center_flat(load_synth(sub, UNTRAINED_MISTRAL))
    bio, times = load_bio(sub)

    sdd = np.sum(d * d, axis=0)
    suu = np.sum(u * u, axis=0)
    sll = np.sum(l * l, axis=0)
    s00 = np.sum(l0 * l0, axis=0)
    sul = np.sum(u * l, axis=0)
    sd0 = np.sum(d * l0, axis=0)
    det_ul = (suu * sll) - (sul * sul)
    det_d0 = (sdd * s00) - (sd0 * sd0)

    d_res_l = residualize_one(d, l, sll)
    l_res_d = residualize_one(l, d, sdd)
    d_res_u = residualize_one(d, u, suu)
    d_res_ul = residualize_two(d, u, l, suu, sll, sul, det_ul)
    l_res_d0 = residualize_two(l, d, l0, sdd, s00, sd0, det_d0)

    uv_d_l_sum = np.zeros((63, bio.shape[-1]), dtype=float)
    uv_l_d_sum = np.zeros_like(uv_d_l_sum)
    uv_d_ul_sum = np.zeros_like(uv_d_l_sum)
    uv_l_d0_sum = np.zeros_like(uv_d_l_sum)
    sv_dl_u_sum = np.zeros_like(uv_d_l_sum)

    rng = np.random.default_rng(SEED + sub)
    n_reps = bio.shape[1]
    for _ in range(ITERATIONS):
        seed_i = int(rng.integers(0, np.iinfo(np.int32).max))
        idx = np.random.RandomState(seed_i).permutation(n_reps)[:n_reps // 2]
        y = center_flat(np.mean(np.delete(bio, idx, axis=1), axis=1))
        n_obs = y.shape[0]

        y_res_l = residualize_one(y, l, sll)
        r2_l = r2_from_residual(y, y_res_l)
        uv_d_l = delta_adj_from_residuals(y, y_res_l, d_res_l, r2_l, n_obs, 1)

        y_res_d = residualize_one(y, d, sdd)
        r2_d = r2_from_residual(y, y_res_d)
        uv_l_d = delta_adj_from_residuals(y, y_res_d, l_res_d, r2_d, n_obs, 1)

        y_res_u = residualize_one(y, u, suu)
        r2_u = r2_from_residual(y, y_res_u)
        uv_d_u = delta_adj_from_residuals(y, y_res_u, d_res_u, r2_u, n_obs, 1)

        y_res_ul = residualize_two(y, u, l, suu, sll, sul, det_ul)
        r2_ul = r2_from_residual(y, y_res_ul)
        uv_d_ul = delta_adj_from_residuals(y, y_res_ul, d_res_ul, r2_ul, n_obs, 2)

        y_res_d0 = residualize_two(y, d, l0, sdd, s00, sd0, det_d0)
        r2_d0 = r2_from_residual(y, y_res_d0)
        uv_l_d0 = delta_adj_from_residuals(y, y_res_d0, l_res_d0, r2_d0, n_obs, 2)

        uv_d_l_sum += uv_d_l.reshape(63, -1)
        uv_l_d_sum += uv_l_d.reshape(63, -1)
        uv_d_ul_sum += uv_d_ul.reshape(63, -1)
        uv_l_d0_sum += uv_l_d0.reshape(63, -1)
        sv_dl_u_sum += (uv_d_u - uv_d_ul).reshape(63, -1)

    return {
        'uv_trained_cornet_given_mistral': uv_d_l_sum / ITERATIONS,
        'uv_mistral_given_trained_cornet': uv_l_d_sum / ITERATIONS,
        'uv_trained_cornet_given_untrained_cornet_mistral': uv_d_ul_sum / ITERATIONS,
        'uv_mistral_given_trained_cornet_untrained_mistral': uv_l_d0_sum / ITERATIONS,
        'sv_trained_cornet_mistral_given_untrained_cornet': sv_dl_u_sum / ITERATIONS,
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


def subject_cache_path(sub):
    return op.join(SUBJECT_CACHE_DIR, f'sub-{sub:02d}.npz')


def load_subject_cache(sub):
    path = subject_cache_path(sub)
    d = np.load(path)
    return {k: d[k] for k in ARRAY_KEYS + ('times',)}


def compute_or_load_subject(sub, overwrite=False):
    os.makedirs(SUBJECT_CACHE_DIR, exist_ok=True)
    path = subject_cache_path(sub)
    if op.exists(path) and op.getsize(path) > 0 and not overwrite:
        print(f'sub-{sub:02d} cached', flush=True)
        return load_subject_cache(sub)
    out = compute_subject(sub)
    np.savez_compressed(path, **out)
    print(f'sub-{sub:02d} computed', flush=True)
    return out


def build_outputs(subjects, overwrite_subjects=False, allow_missing=False):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    arrays = {k: [] for k in ARRAY_KEYS}
    times = None
    done_subjects = []
    for sub in subjects:
        try:
            out = compute_or_load_subject(sub, overwrite_subjects)
        except Exception as exc:
            if not allow_missing:
                raise
            print(f'sub-{sub:02d} skipped: {exc}', flush=True)
            continue
        for k in arrays:
            arrays[k].append(out[k])
        done_subjects.append(sub)
        if times is None:
            times = out['times']

    if not done_subjects:
        raise RuntimeError('no subjects available for Mistral figure')

    for k in arrays:
        arrays[k] = np.stack(arrays[k])
    arrays['times'] = times
    arrays['subjects'] = np.asarray(done_subjects, dtype=int)

    np.savez_compressed(
        op.join(ANALYSIS_DIR, 'conditional_sv_trained_cornet_mistral_arrays.npz'),
        **arrays,
    )

    curves = {
        'UV(trained CORnet-S | Mistral)':
            arrays['uv_trained_cornet_given_mistral'].mean(axis=1),
        'UV(Mistral | trained CORnet-S)':
            arrays['uv_mistral_given_trained_cornet'].mean(axis=1),
        'UV(trained CORnet-S | untrained CORnet-S + Mistral)':
            arrays['uv_trained_cornet_given_untrained_cornet_mistral'].mean(axis=1),
        'UV(Mistral | trained CORnet-S + untrained Mistral)':
            arrays['uv_mistral_given_trained_cornet_untrained_mistral'].mean(axis=1),
        'SV(trained CORnet-S ∩ Mistral | untrained CORnet-S)':
            arrays['sv_trained_cornet_mistral_given_untrained_cornet'].mean(axis=1),
    }
    rows = [summarize(label, curve, times) for label, curve in curves.items()]

    csv_path = op.join(ANALYSIS_DIR, 'conditional_sv_trained_cornet_mistral_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    styles = {
        'UV(trained CORnet-S | Mistral)': ('#1f77b4', '-'),
        'UV(Mistral | trained CORnet-S)': ('#ff7f0e', '-'),
        'UV(trained CORnet-S | untrained CORnet-S + Mistral)': ('#08519c', '-'),
        'UV(Mistral | trained CORnet-S + untrained Mistral)': ('#c65102', '-'),
        'SV(trained CORnet-S ∩ Mistral | untrained CORnet-S)': ('#6a3d9a', '-'),
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
    ax.set_title('Conditional partition: trained CORnet-S and e5-Mistral')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(times[0], times[-1])
    fig.tight_layout()
    fig.savefig(op.join(OUT_DIR, 'conditional_sv_trained_cornet_mistral.png'), dpi=200)
    plt.close(fig)

    print('=== Conditional trained CORnet-S / e5-Mistral partition ===')
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
    print(' ', op.join(ANALYSIS_DIR, 'conditional_sv_trained_cornet_mistral_arrays.npz'))
    print(' ', op.join(OUT_DIR, 'conditional_sv_trained_cornet_mistral.png'))


def parse_subjects(text):
    subjects = []
    for part in text.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-', 1)
            subjects.extend(range(int(lo), int(hi) + 1))
        elif part:
            subjects.append(int(part))
    return subjects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subjects', default='1-10')
    parser.add_argument('--overwrite-subjects', action='store_true')
    parser.add_argument('--allow-missing', action='store_true')
    args = parser.parse_args()
    build_outputs(
        parse_subjects(args.subjects),
        overwrite_subjects=args.overwrite_subjects,
        allow_missing=args.allow_missing,
    )


if __name__ == '__main__':
    main()
