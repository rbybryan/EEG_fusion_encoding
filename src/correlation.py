import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import argparse
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

    # Adjust synthetic data dimensions
    synt_test = synt_test.reshape(200, -1, 100)[:, :63, :]
    bio_test = bio_test[:, :, :63, :]

    # Initialize result matrices
    iterations, channels, time_points = 100, bio_test.shape[2], bio_test.shape[3]
    correlation_end = np.zeros((iterations, channels, time_points))
    noise_ceiling_low = np.zeros((iterations, channels, time_points))
    noise_ceiling_up = np.zeros((iterations, channels, time_points))

    # Average across all repetitions for upper bound noise ceiling
    bio_data_avg_all = np.mean(bio_test, axis=1)

    # Compute correlation
    for i in tqdm.tqdm(range(iterations), desc="Computing correlations"):
        shuffle_idx = resample(np.arange(bio_test.shape[1]), replace=False)[:bio_test.shape[1] // 2]
        bio_data_avg_half_1 = np.mean(np.delete(bio_test, shuffle_idx, axis=1), axis=1)
        bio_data_avg_half_2 = np.mean(bio_test[:, shuffle_idx, :, :], axis=1)

        for t in range(time_points):
            if np.sum(synt_test[:, :, t]) != 0:
                for c in range(channels):
                    correlation_end[i, c, t] = np.corrcoef(synt_test[:, c, t], bio_data_avg_half_1[:, c, t])[0, 1]
                    noise_ceiling_low[i, c, t] = np.corrcoef(bio_data_avg_half_2[:, c, t], bio_data_avg_half_1[:, c, t])[0, 1]
                    noise_ceiling_up[i, c, t] = np.corrcoef(bio_data_avg_all[:, c, t], bio_data_avg_half_1[:, c, t])[0, 1]

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
    plt.show()
    plt.close()


# Argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--sub', type=int, default=4, help='Subject identifier')
parser.add_argument('--project_dir', type=str, default='/scratch/byrong/encoding/data', help='Root project directory')
parser.add_argument('--data_path_bio', type=str, default='/scratch/byrong/encoding/data/eeg_dataset/preprocessed_eeg_data_v1')
parser.add_argument('--file', type=str, required=True, help='Synthetic EEG data file name')

args = parser.parse_args()

model_name = args.file
args.file += '.npy'

bio_file = os.path.join(args.data_path_bio, f'eeg_sub-{args.sub:02}_split-test.npy')
bio_data = np.load(bio_file, allow_pickle=True).item()

bio_test = bio_data['preprocessed_eeg_data']
ch_names = bio_data['ch_names']
times = bio_data['times']

synt_file = os.path.join(args.project_dir, 'linear_results', 'sub-'+
              format(args.sub,'02'), 'synthetic_eeg_data', args.file)
synt_data = np.load(synt_file, allow_pickle=True).item()

synt_test = synt_data['synthetic_data']

correlation_file = os.path.join(args.project_dir, 'linear_results', f'sub-{args.sub:02}', 'correlation', f'correlation_{model_name}.npy')

if not os.path.exists(correlation_file):
    plot_correlation(synt_test, bio_test, ch_names, times, model_name, args.sub, os.path.join(args.project_dir, 'linear_results'))
