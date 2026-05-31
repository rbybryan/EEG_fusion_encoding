"""Cluster-based permutation testing for 1D timecourses.

Wraps `mne.stats.permutation_cluster_1samp_test` for the rebuttal figures.

The standard EEG temporal-cluster test (Maris & Oostenveld 2007):
1. Compute t-stat at each timepoint across subjects.
2. Form clusters where t exceeds a threshold (default: p<0.05 uncorrected,
   df = n_sub-1, one- or two-tailed depending on `tail`).
3. Cluster-level statistic = sum of t-values inside each cluster.
4. Build the null by sign-flipping subjects' data (one-sample) and
   recomputing the max cluster stat across n_permutations.
5. Cluster p = (#perms with max stat >= observed) / (n_perm + 1).

Use `tail=1` for "trace > 0" (one-tailed greater), `tail=0` for two-tailed.
"""

import numpy as np
from mne.stats import permutation_cluster_1samp_test


def cluster_perm_1samp_sig(arr, alpha=0.05, tail=1, n_permutations=10000,
                            seed=42, threshold=None):
    """One-sample cluster-perm test on (n_subjects, n_times).

    Returns
    -------
    sig : bool array of shape (n_times,)
        True at timepoints inside any cluster with p < alpha.
    """
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f'expected (n_subjects, n_times); got {arr.shape}')
    n_t = arr.shape[1]
    if arr.shape[0] < 3 or np.allclose(np.nan_to_num(arr).std(axis=0), 0):
        return np.zeros(n_t, dtype=bool)
    arr = np.nan_to_num(arr)
    _, clusters, cluster_pvals, _ = permutation_cluster_1samp_test(
        arr,
        n_permutations=n_permutations,
        threshold=threshold,
        tail=tail,
        out_type='mask',
        verbose=False,
        seed=seed,
    )
    sig = np.zeros(n_t, dtype=bool)
    for cl, p in zip(clusters, cluster_pvals):
        if p >= alpha:
            continue
        # mne returns tuples of slices (or arrays of indices) for 1D data
        # even when out_type='mask'; normalise to a boolean mask.
        if isinstance(cl, tuple):
            mask = np.zeros(n_t, dtype=bool)
            mask[cl[0]] = True
        else:
            mask = np.asarray(cl)
            if mask.dtype != bool:
                idx = mask
                mask = np.zeros(n_t, dtype=bool)
                mask[idx] = True
        sig |= mask
    return sig


def add_cluster_sig_bar(ax, arr, times, color, y, *, tail=1, alpha=0.05,
                        n_permutations=10000, marker_size=4):
    """Compute cluster-perm sig and scatter markers at height `y`."""
    sig = cluster_perm_1samp_sig(
        arr, alpha=alpha, tail=tail, n_permutations=n_permutations
    )
    if sig.any():
        ax.scatter(times[sig], np.full(sig.sum(), y), s=marker_size,
                   marker='s', color=color, edgecolors='none')
    return sig
