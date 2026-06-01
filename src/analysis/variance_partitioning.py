"""Variance partitioning between two encoding models (unique and shared variance).

Given the held-out synthetic-EEG predictions of two models (A and B) and of their
fusion model (A+B), this computes the standard Borcard-Legendre three-way
partition of explained variance against the recorded EEG, time-resolved per
channel and time point:

    R2(A)   = R2( y ~ pred_A )
    R2(B)   = R2( y ~ pred_B )
    R2(AB)  = R2( y ~ pred_AB )          (fusion model prediction)

    SV(A, B) = R2(A) + R2(B) - R2(AB)    shared variance (A intersect B)
    UV(A|B)  = R2(AB) - R2(B)            variance unique to A
    UV(B|A)  = R2(AB) - R2(A)            variance unique to B

Each R2 is the squared Pearson correlation between a model prediction and a
split-half average of the recorded EEG, computed independently per channel and
time point and averaged over split-half resampling iterations (the same
convention as ``correlation.py`` / ``partial_correlation.py``). This single
script replaces the model-specific shared-variance scripts: pass any pair of
models plus their fusion via ``--pred_a / --pred_b / --pred_ab``.

Inputs (per subject, under ``<project_dir>/linear_results``):
    synthetic predictions  linear_results/sub-XX/synthetic_eeg_data/<name>[_all].npy
                           key 'synthetic_data', shape (images, channels, time)
    recorded EEG           eeg_dataset/preprocessed_eeg_data_v1/eeg_sub-XX_split-test.npy
                           key 'preprocessed_eeg_data', shape (images, reps, channels, time)

Outputs (under ``<repo>/analysis``):
    variance_partitioning_<tag>_summary.csv   per-subject UV/SV window means and peaks
    variance_partitioning_<tag>_arrays.npz    full (subject, channel, time) UV/SV arrays

Example
-------
    python variance_partitioning.py \\
        --pred_a cornet_s_r2_v1 \\
        --pred_b text_embedding_large_r2_v1 \\
        --pred_ab cornet_s_with_text_embedding_large_r2_v1 \\
        --tag cornet_tel
"""

import argparse
import csv
import os
import os.path as op

import numpy as np
from sklearn.utils import resample

# --- path configuration (portable defaults for public release) ---
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DATA_ROOT = os.environ.get('EEG_FUSION_DATA', 'data')
# --- end path configuration ---

ANALYSIS_DIR = op.join(_REPO, 'analysis')
N_CHANNELS = 63
SUBJECTS = list(range(1, 11))


def load_synth(project_dir, sub, name):
    """Load a held-out synthetic-EEG prediction, shape (images, channels, time)."""
    base = op.join(project_dir, 'linear_results', f'sub-{sub:02d}',
                   'synthetic_eeg_data')
    for cand in (f'{name}.npy', f'{name}_all.npy'):
        path = op.join(base, cand)
        if op.exists(path):
            return np.load(path, allow_pickle=True).item()['synthetic_data']
    raise FileNotFoundError(f'no synthetic prediction file for {name!r} under {base}')


def load_bio(project_dir, sub):
    """Load the recorded test EEG and its time axis.

    Returns
    -------
    eeg : numpy.ndarray
        Shape (images, repetitions, channels, time).
    times : numpy.ndarray
        Time axis in seconds.
    """
    path = op.join(project_dir, 'eeg_dataset', 'preprocessed_eeg_data_v1',
                   f'eeg_sub-{sub:02d}_split-test.npy')
    d = np.load(path, allow_pickle=True).item()
    return d['preprocessed_eeg_data'], np.asarray(d['times'])


def r2_single(pred, bio_half):
    """Vectorised squared Pearson r between a prediction and the target.

    Parameters
    ----------
    pred, bio_half : numpy.ndarray
        Arrays of shape (images, channels, time); correlation is computed across
        images independently for each (channel, time).

    Returns
    -------
    numpy.ndarray
        R2 of shape (channels, time).
    """
    pred_c = pred - pred.mean(axis=0, keepdims=True)
    bio_c = bio_half - bio_half.mean(axis=0, keepdims=True)
    num = np.sum(pred_c * bio_c, axis=0) ** 2
    den = np.sum(pred_c ** 2, axis=0) * np.sum(bio_c ** 2, axis=0)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def partition_subject(project_dir, sub, name_a, name_b, name_ab, iterations, seed):
    """Compute the three-way partition for one subject, averaged over resamples.

    Returns a dict with UV(A|B), UV(B|A), SV(A,B) and the component R2 maps,
    each of shape (channels, time), plus the time axis.
    """
    pred_a = load_synth(project_dir, sub, name_a)[:, :N_CHANNELS, :].astype(float)
    pred_b = load_synth(project_dir, sub, name_b)[:, :N_CHANNELS, :].astype(float)
    pred_ab = load_synth(project_dir, sub, name_ab)[:, :N_CHANNELS, :].astype(float)
    bio, times = load_bio(project_dir, sub)
    bio = bio[:, :, :N_CHANNELS, :]
    n_reps = bio.shape[1]

    rng = np.random.default_rng(seed + sub)
    acc = {'r2_a': [], 'r2_b': [], 'r2_ab': []}
    for _ in range(iterations):
        # Split-half target: average the held-in half of the EEG repetitions.
        idx = resample(np.arange(n_reps), replace=False,
                       random_state=int(rng.integers(0, np.iinfo(np.int32).max)))[:n_reps // 2]
        bio_half = np.mean(np.delete(bio, idx, axis=1), axis=1)
        acc['r2_a'].append(r2_single(pred_a, bio_half))
        acc['r2_b'].append(r2_single(pred_b, bio_half))
        acc['r2_ab'].append(r2_single(pred_ab, bio_half))

    r2_a = np.mean(acc['r2_a'], axis=0)
    r2_b = np.mean(acc['r2_b'], axis=0)
    r2_ab = np.mean(acc['r2_ab'], axis=0)
    return {
        'uv_a': r2_ab - r2_b,
        'uv_b': r2_ab - r2_a,
        'sv': r2_a + r2_b - r2_ab,
        'r2_a': r2_a,
        'r2_b': r2_b,
        'r2_ab': r2_ab,
        'times': times,
    }


def window_mean(mat, times, lo, hi):
    """Mean of a (channel, time) map over a time window, all channels."""
    mask = (times >= lo) & (times <= hi)
    if not np.any(mask):
        return float('nan')
    return float(mat[:, mask].mean())


def summarize(term, curve, times):
    """Build a CSV summary row for one partition term's channel-mean curve."""
    peak_idx = int(np.argmax(curve))
    return {
        'term': term,
        'mean_all_time': f'{float(curve.mean()):.8f}',
        'peak': f'{float(curve[peak_idx]):.8f}',
        'peak_time_s': f'{float(times[peak_idx]):.6f}',
        'mean_0_200ms': f'{window_mean(curve[None, :], times, 0.0, 0.2):.8f}',
        'mean_200_600ms': f'{window_mean(curve[None, :], times, 0.2, 0.6):.8f}',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pred_a', required=True,
                        help='Synthetic-prediction basename for model A.')
    parser.add_argument('--pred_b', required=True,
                        help='Synthetic-prediction basename for model B.')
    parser.add_argument('--pred_ab', required=True,
                        help='Synthetic-prediction basename for the A+B fusion model.')
    parser.add_argument('--tag', required=True,
                        help='Short label used in the output file names.')
    parser.add_argument('--project_dir', default=_DATA_ROOT,
                        help='Root data directory holding linear_results and eeg_dataset.')
    parser.add_argument('--subjects', default=None,
                        help="Comma/range list of subjects, e.g. '1-10' or '1,3,5' "
                             '(default: 1-10).')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of split-half resampling iterations.')
    parser.add_argument('--seed', type=int, default=20260510,
                        help='Base random seed (per-subject seed is seed + subject).')
    args = parser.parse_args()

    subjects = parse_subjects(args.subjects) if args.subjects else SUBJECTS
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    terms = ('uv_a', 'uv_b', 'sv')
    per_subject = {t: [] for t in terms}
    components = {c: [] for c in ('r2_a', 'r2_b', 'r2_ab')}
    times = None
    for sub in subjects:
        print(f'subject {sub:02d}', flush=True)
        out = partition_subject(args.project_dir, sub, args.pred_a, args.pred_b,
                                args.pred_ab, args.iterations, args.seed)
        for t in terms:
            per_subject[t].append(out[t])
        for c in components:
            components[c].append(out[c])
        if times is None:
            times = out['times']

    arrays = {'times': times, 'subjects': np.asarray(subjects)}
    rows = []
    pretty = {'uv_a': f'UV({args.pred_a} | {args.pred_b})',
              'uv_b': f'UV({args.pred_b} | {args.pred_a})',
              'sv': f'SV({args.pred_a}, {args.pred_b})'}
    for t in terms:
        stacked = np.stack(per_subject[t])          # (subjects, channels, time)
        arrays[t] = stacked
        curve = stacked.mean(axis=(0, 1))           # channel- and subject-mean
        rows.append(summarize(pretty[t], curve, times))
    for c in components:
        arrays[c] = np.stack(components[c])

    summary_path = op.join(ANALYSIS_DIR, f'variance_partitioning_{args.tag}_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    arrays_path = op.join(ANALYSIS_DIR, f'variance_partitioning_{args.tag}_arrays.npz')
    np.savez_compressed(arrays_path, **arrays)

    print(f'wrote {summary_path}')
    print(f'wrote {arrays_path}')


def parse_subjects(spec):
    """Parse a subject spec like '1-10' or '1,3,5' into a list of ints."""
    subjects = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-')
            subjects.extend(range(int(lo), int(hi) + 1))
        elif part:
            subjects.append(int(part))
    return subjects


if __name__ == '__main__':
    main()
