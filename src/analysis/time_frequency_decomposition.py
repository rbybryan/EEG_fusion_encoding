"""Morlet time-frequency decomposition of the preprocessed EEG.

Computes a complex Morlet-wavelet transform of the recorded EEG for one subject
and writes, per frequency, the complex coefficients, the log power, and the
unit-magnitude phase. These time-frequency representations are the input for
band-resolved encoding/decoding analyses (e.g. fitting encoding models within
delta/theta/alpha bands).

For each frequency the transform is computed with ``mne.time_frequency``'s
array Morlet routine using log-spaced frequencies and a frequency-dependent
number of cycles (``freq / 3``, capped at 5). Training-set repetitions are
processed one at a time and averaged to limit peak memory.

Input (under ``<project_dir>/eeg_dataset/preprocessed_eeg_data_v1``):
    eeg_sub-XX_split-train.npy   key 'preprocessed_eeg_data',
                                 shape (images, repetitions, channels, time)

Output (under ``<project_dir>/eeg_dataset/tf_decomposition/...``):
    tf_<split>_power_<j>.npy     log power for frequency index j
    tf_<split>_complex_<j>.npy   complex coefficients for frequency index j
    tf_<split>_phase_<j>.npy     unit-magnitude phase for frequency index j

Example
-------
    python time_frequency_decomposition.py --sub 1 --project_dir /path/to/encoding/data
"""

import argparse
import os
import os.path as op

import mne
import numpy as np

SEED = 20200220
N_CHANNELS = 63


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sub', default=1, type=int, help='Subject identifier.')
    parser.add_argument('--project_dir', default=os.environ.get('EEG_FUSION_DATA', 'data'),
                        type=str, help='Root data directory.')
    parser.add_argument('--split', default='train', choices=['train', 'test'],
                        help='Which EEG partition to decompose.')
    parser.add_argument('--n_freqs', default=30, type=int,
                        help='Number of log-spaced frequencies.')
    parser.add_argument('--low', default=2.0, type=float,
                        help='Lowest frequency (Hz).')
    parser.add_argument('--high', default=70.0, type=float,
                        help='Highest frequency (Hz).')
    parser.add_argument('--sfreq', default=200, type=int,
                        help='Sampling frequency of the preprocessed EEG (Hz).')
    parser.add_argument('--n_jobs', default=16, type=int,
                        help='Parallel jobs for the Morlet transform.')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(SEED)

    data_dir = op.join(args.project_dir, 'eeg_dataset', 'preprocessed_eeg_data_v1')
    eeg_file = op.join(data_dir, f'eeg_sub-{args.sub:02d}_split-{args.split}.npy')
    data = np.load(eeg_file, allow_pickle=True).item()
    eeg = data['preprocessed_eeg_data'][:, :, :N_CHANNELS, :]   # (img, rep, ch, time)
    ch_names = list(data['ch_names'][:N_CHANNELS])
    n_reps = eeg.shape[1]

    save_dir = op.join(args.project_dir, 'eeg_dataset', 'tf_decomposition',
                       'preprocessed_eeg_data_v1', f'sub-{args.sub:02d}')
    os.makedirs(save_dir, exist_ok=True)

    info = mne.create_info(ch_names=ch_names, sfreq=args.sfreq, ch_types='eeg')  # noqa: F841
    frequencies = np.logspace(np.log10(args.low), np.log10(args.high), args.n_freqs)
    n_cycles = frequencies / 3.0
    n_cycles[n_cycles > 5] = 5

    for j, freq in enumerate(frequencies):
        phase_path = op.join(save_dir, f'tf_{args.split}_phase_{j}.npy')
        if op.exists(phase_path):
            continue

        # Average the Morlet transform over repetitions, one repetition at a time.
        complex_sum = np.zeros(eeg[:, 0].shape, dtype=complex)
        power_sum = np.zeros(eeg[:, 0].shape, dtype=float)
        phase_sum = np.zeros(eeg[:, 0].shape, dtype=complex)
        for rep in range(n_reps):
            tfr = mne.time_frequency.tfr_array_morlet(
                eeg[:, rep], sfreq=args.sfreq, freqs=frequencies[j:j + 1],
                n_cycles=n_cycles[j:j + 1], decim=1, n_jobs=args.n_jobs)[:, :, 0]
            complex_sum += tfr
            magnitude = np.abs(tfr)
            power_sum += 10 * np.log10(magnitude ** 2 + 1e-12)
            magnitude[magnitude == 0] = 1
            phase_sum += tfr / magnitude

        np.save(op.join(save_dir, f'tf_{args.split}_power_{j}.npy'), power_sum / n_reps)
        np.save(op.join(save_dir, f'tf_{args.split}_complex_{j}.npy'), complex_sum / n_reps)
        phase = phase_sum / n_reps
        phase = phase / np.abs(phase)
        np.save(phase_path, phase)
        print(f'frequency {j + 1}/{args.n_freqs} ({freq:.1f} Hz) done', flush=True)

    print(f'time-frequency decomposition written to {save_dir}')


if __name__ == '__main__':
    main()
