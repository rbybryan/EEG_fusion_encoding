"""Compute and plot EEG encoding-model correlations and noise ceilings.

This script loads biological (test) EEG data and synthetic EEG data produced by
a linearizing encoding model, computes the per-channel/per-time-point Pearson
correlation between them together with lower- and upper-bound noise ceilings,
saves the results, and produces a summary correlation plot.

The correlation is vectorised over channels and time. The split-half resampling
is drawn in the identical order, and time points whose synthetic data are all
zero are left at zero, so the averaged results match the original per-cell
``np.corrcoef`` loop while running substantially faster.
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tqdm
from sklearn.utils import resample


def plot_correlation(synt_test, bio_test, ch_names, times, model_name, sub, data_path):
    """Compute correlations and noise ceilings and save results.

    Parameters
    ----------
    synt_test : ndarray
        Synthetic EEG data (conditions × channels × time points).
    bio_test : ndarray
        Biological EEG data (conditions × repetitions × channels × time points).
    ch_names : list
        List of EEG channel names.
    times : ndarray
        EEG time points.
    model_name : str
        Name of the model used.
    sub : int
        Subject identifier.
    data_path : str
        Path to save the correlation results.
    """
    # Match the manuscript correlation pipeline's deterministic split-half draws.
    np.random.seed(seed=20200220)

    # Adjust synthetic data dimensions
    synt_test = synt_test[:, :63, :]
    bio_test = bio_test[:, :, :63, :]

    # Initialize result matrices
    iterations, channels, time_points = 100, bio_test.shape[2], bio_test.shape[3]
    correlation_end = np.zeros((iterations, channels, time_points))
    noise_ceiling_low = np.zeros((iterations, channels, time_points))
    noise_ceiling_up = np.zeros((iterations, channels, time_points))

    # Average across all repetitions for upper bound noise ceiling
    bio_data_avg_all = np.mean(bio_test, axis=1)

    # Time points with all-zero synthetic data are left at zero, reproducing the
    # original per-cell loop which skipped them via `if np.sum(...) != 0`.
    time_mask = synt_test.sum(axis=(0, 1)) != 0   # (time_points,)

    def _pearson_over_conditions(a, b):
        """Pearson r along the condition axis (axis 0); returns (channels, time)."""
        a = a - a.mean(axis=0, keepdims=True)
        b = b - b.mean(axis=0, keepdims=True)
        num = (a * b).sum(axis=0)
        den = np.sqrt((a ** 2).sum(axis=0) * (b ** 2).sum(axis=0))
        with np.errstate(divide='ignore', invalid='ignore'):
            return num / den

    # Compute correlation. Vectorised over channels and time; the split-half
    # resampling is drawn in the identical order as before, so the averaged
    # results match the original per-cell np.corrcoef loop.
    for i in tqdm.tqdm(range(iterations), desc="Computing correlations"):
        shuffle_idx = resample(np.arange(bio_test.shape[1]), replace=False)[:bio_test.shape[1] // 2]
        bio_data_avg_half_1 = np.mean(np.delete(bio_test, shuffle_idx, axis=1), axis=1)
        bio_data_avg_half_2 = np.mean(bio_test[:, shuffle_idx, :, :], axis=1)

        corr = _pearson_over_conditions(synt_test, bio_data_avg_half_1)
        ncl = _pearson_over_conditions(bio_data_avg_half_2, bio_data_avg_half_1)
        ncu = _pearson_over_conditions(bio_data_avg_all, bio_data_avg_half_1)

        correlation_end[i] = np.where(time_mask[None, :], corr, 0.0)
        noise_ceiling_low[i] = np.where(time_mask[None, :], ncl, 0.0)
        noise_ceiling_up[i] = np.where(time_mask[None, :], ncu, 0.0)

    # Average results across iterations
    correlation = correlation_end.mean(axis=0)
    noise_ceiling_low = noise_ceiling_low.mean(axis=0)
    noise_ceiling_up = noise_ceiling_up.mean(axis=0)

    # Save results
    results_dict = {
        'correlation': correlation,
        'noise_ceiling_low': noise_ceiling_low,
        'noise_ceiling_up': noise_ceiling_up,
        'times': times,
        'ch_names': ch_names
    }

    save_dir = os.path.join(data_path, f'sub-{sub:02}', 'correlation')
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f'correlation_{model_name}.npy'), results_dict)

    # Plotting
    plt.plot(times, correlation.mean(axis=0), label=model_name)
    plt.xlabel("Time (s)")
    plt.ylabel("Correlation")
    plt.legend()
    plt.title(f'Subject {sub:02}')
    plot_path = os.path.join(save_dir, f'correlation_{model_name}.png')
    plt.savefig(plot_path)
    plt.close()


# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, default=4, help='Subject identifier')
parser.add_argument('--project_dir', type=str,
                    default=os.environ.get('EEG_FUSION_DATA', 'data'),
                    help='Root project directory')
parser.add_argument('--data_path_bio', type=str,
                    default=os.path.join(os.environ.get('EEG_FUSION_DATA', 'data'),
                                         'eeg_dataset', 'preprocessed_eeg_data_v1'),
                    help='Directory holding the preprocessed test EEG')
parser.add_argument('--file', type=str, required=True, help='Synthetic EEG data file name')

args = parser.parse_args()

model_name = args.file
args.file += '.npy'

bio_file = os.path.join(args.data_path_bio, f'eeg_sub-{args.sub:02}_split-test.npy')
bio_data = np.load(bio_file, allow_pickle=True).item()

bio_test = bio_data['preprocessed_eeg_data']
ch_names = bio_data['ch_names']
times = bio_data['times']

synt_file = os.path.join(args.project_dir, 'linear_results', 'sub-' +
                         format(args.sub, '02'), 'synthetic_eeg_data', args.file)
synt_data = np.load(synt_file, allow_pickle=True).item()

synt_test = synt_data['synthetic_data']

correlation_file = os.path.join(args.project_dir, 'linear_results', f'sub-{args.sub:02}',
                                'correlation', f'correlation_{model_name}.npy')

if not os.path.exists(correlation_file):
    plot_correlation(synt_test, bio_test, ch_names, times, model_name, args.sub,
                     os.path.join(args.project_dir, 'linear_results'))
