"""Partial correlation between a predictor model and biological EEG,
controlling for a second model's predictions.

Adapted from EEG_fusion_encoding/src/partial_correlation.py for the
rebuttal pipeline (already-aggregated `*_all.npy` synthetic outputs of
shape (200, 63, 180)).

partial_corr(predictor=p, outcome=bio | control=c) answers:
    "How much variance does p explain in bio that c does NOT explain?"

Per-iteration the bio target is the held-in-half average across reps,
matching `src/correlation.py`.
"""

import argparse
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
import tqdm
from sklearn.utils import resample


def partial_corr_lstsq(x, y, control):
    """Compute the partial correlation ``corr(x, y | control)``.

    The predictor ``x`` and outcome ``y`` are each linearly regressed on
    ``control`` (with an intercept) and the Pearson correlation of the
    residuals is returned.

    Parameters
    ----------
    x : array_like
        Predictor values, one per sample.
    y : array_like
        Outcome values, one per sample.
    control : array_like
        Variable to partial out, one value per sample.

    Returns
    -------
    float
        Partial correlation of ``x`` and ``y`` controlling for
        ``control``. Returns ``0.0`` if either residual vector has zero
        standard deviation.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    control = np.asarray(control)
    ones = np.ones((control.shape[0], 1))
    X = np.hstack([ones, control.reshape(-1, 1)])
    beta_x, _, _, _ = np.linalg.lstsq(X, x, rcond=None)
    beta_y, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    rx = x - X @ beta_x
    ry = y - X @ beta_y
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def compute_partial_correlation(pred_test, ctrl_test, bio_test, iterations=100):
    """Compute per-channel, per-timepoint partial correlation against EEG.

    For each bootstrap iteration a random held-in half of the repetitions
    is averaged to form the biological target, and the partial correlation
    ``corr(pred, bio | ctrl)`` is computed for every channel and timepoint.

    Parameters
    ----------
    pred_test : numpy.ndarray
        Predictor synthetic EEG of shape ``(n_images, n_channels, n_times)``.
    ctrl_test : numpy.ndarray
        Control synthetic EEG of shape ``(n_images, n_channels, n_times)``.
    bio_test : numpy.ndarray
        Biological EEG of shape ``(n_images, n_reps, n_channels, n_times)``.
    iterations : int, optional
        Number of bootstrap iterations, by default 100.

    Returns
    -------
    numpy.ndarray
        Mean partial correlation over iterations, shape
        ``(n_channels, n_times)``.
    """
    pred_test = pred_test[:, :63, :]
    ctrl_test = ctrl_test[:, :63, :]
    bio_test = bio_test[:, :, :63, :]
    n_ch = bio_test.shape[2]
    n_t = bio_test.shape[3]
    out = np.zeros((iterations, n_ch, n_t))
    for i in tqdm.tqdm(range(iterations), desc='partial corr'):
        idx = resample(np.arange(bio_test.shape[1]), replace=False)[
            : bio_test.shape[1] // 2
        ]
        bio_half = np.mean(np.delete(bio_test, idx, axis=1), axis=1)
        for t in range(n_t):
            if np.sum(pred_test[:, :, t]) == 0:
                continue
            for c in range(n_ch):
                try:
                    out[i, c, t] = partial_corr_lstsq(
                        pred_test[:, c, t],
                        bio_half[:, c, t],
                        ctrl_test[:, c, t],
                    )
                except (np.linalg.LinAlgError, ValueError):
                    out[i, c, t] = 0.0
    return np.nanmean(out, axis=0)


def load_synth(project_dir, sub, name):
    """Load synthetic EEG data; auto-appends `_all` if needed."""
    base = op.join(project_dir, 'linear_results', f'sub-{sub:02d}',
                   'synthetic_eeg_data')
    for cand in (f'{name}.npy', f'{name}_all.npy'):
        path = op.join(base, cand)
        if op.exists(path):
            return np.load(path, allow_pickle=True).item()['synthetic_data']
    raise FileNotFoundError(f'no synth file for {name} under {base}')


def load_bio(project_dir, sub):
    """Load preprocessed biological test-split EEG for one subject.

    Returns
    -------
    tuple
        ``(preprocessed_eeg_data, ch_names, times)``.
    """
    path = op.join(project_dir, 'eeg_dataset', 'preprocessed_eeg_data_v1',
                   f'eeg_sub-{sub:02d}_split-test.npy')
    d = np.load(path, allow_pickle=True).item()
    return d['preprocessed_eeg_data'], d['ch_names'], d['times']


def main():
    """Parse CLI arguments, compute partial correlation, and save results."""
    p = argparse.ArgumentParser()
    p.add_argument('--sub', type=int, required=True,
                   help='Subject number')
    p.add_argument('--project_dir', type=str,
                   default=_DATA_ROOT,
                   help='Project data root directory')
    p.add_argument('--predictor', type=str, required=True,
                   help='Predictor synthetic file basename (no .npy)')
    p.add_argument('--control', type=str, required=True,
                   help='Control synthetic file basename (no .npy)')
    p.add_argument('--label', type=str, required=True,
                   help='Short label for the control/contrast (used in filename)')
    p.add_argument('--iterations', type=int, default=100,
                   help='Number of bootstrap iterations')
    p.add_argument('--overwrite', action='store_true',
                   help='Recompute and overwrite an existing output file')
    args = p.parse_args()

    save_dir = op.join(args.project_dir, 'linear_results',
                       f'sub-{args.sub:02d}', 'correlation')
    os.makedirs(save_dir, exist_ok=True)
    out_path = op.join(save_dir, f'partial_correlation_{args.predictor}__given__{args.label}.npy')
    if op.exists(out_path) and op.getsize(out_path) > 0 and not args.overwrite:
        print(f'exists, skipping: {out_path}')
        return

    bio, ch_names, times = load_bio(args.project_dir, args.sub)
    pred = load_synth(args.project_dir, args.sub, args.predictor)
    ctrl = load_synth(args.project_dir, args.sub, args.control)
    pcorr = compute_partial_correlation(pred, ctrl, bio, args.iterations)

    np.save(out_path, {
        'partial_correlation': pcorr,
        'predictor': args.predictor,
        'control': args.control,
        'label': args.label,
        'times': times,
        'ch_names': ch_names,
    })

    plt.figure(figsize=(8, 4))
    plt.plot(times, pcorr.mean(axis=0))
    plt.xlabel('Time (s)')
    plt.ylabel(f'partial r ({args.predictor} | {args.label})')
    plt.title(f'sub-{args.sub:02d}')
    plt.axhline(0, color='k', lw=0.5)
    plt.tight_layout()
    plt.savefig(out_path.replace('.npy', '.png'), dpi=150)
    plt.close()
    print(f'saved: {out_path}')


if __name__ == '__main__':
    main()
