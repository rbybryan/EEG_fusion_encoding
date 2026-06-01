"""Preprocess the raw EEG data:
	- channel selection
	- filtering,
	- epoching,
	- current source density transform,
	- frequency downsampling,
	- baseline correction,
	- zscoring of each recording sessions.

After preprocessing, the EEG data is reshaped to:
(Image conditions x EEG repetitions x EEG channels x EEG time points).

The data of the test and training EEG partitions is saved independently.

Parameters
----------
sub : int
	Used subject.
n_ses : int
	Number of EEG sessions.
lowpass : float
	Lowpass filter frequency.
highpass : float
	Highpass filter frequency.
tmin : float
	Start time of the epochs in seconds, relative to stimulus onset.
tmax : float
	End time of the epochs in seconds, relative to stimulus onset.
baseline_correction : int
	Whether to baseline correct [1] or not [0] the data.
baseline_mode : str
	Whether to apply 'mean' or 'zscore' baseline correction mode.
csd : int
	Whether to transform the data into current source density [1] or not [0].
sfreq : int
	Downsampling frequency.
mvnn : str
	Whether to compute the MVNN covariace matrices for each time point
	('time') or for each epoch/repetition ('epochs').
things_eeg_2_dir : str
	Directory of the THINGS EEG2 dataset.
project_dir : str
	Directory of the project folder.

"""

import os
import argparse

from preprocessing_utils import epoching
from preprocessing_utils import save_prepr
from preprocessing_utils import zscore


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int,
                    help='Used subject.')
parser.add_argument('--n_ses', default=4, type=int,
                    help='Number of EEG sessions.')
parser.add_argument('--lowpass', default=100, type=float,
                    help='Lowpass filter frequency.')
parser.add_argument('--highpass', default=0.03, type=float,
                    help='Highpass filter frequency.')
parser.add_argument('--tmin', default=-.1, type=float,
                    help='Start time of the epochs in seconds, relative to '
                         'stimulus onset.')
parser.add_argument('--tmax', default=.8, type=float,
                    help='End time of the epochs in seconds, relative to '
                         'stimulus onset.')
parser.add_argument('--baseline_correction', default=1, type=int,
                    help='Whether to baseline correct [1] or not [0] the data.')
parser.add_argument('--baseline_mode', default='zscore', type=str,
                    help="Whether to apply 'mean' or 'zscore' baseline "
                         "correction mode.")
parser.add_argument('--csd', default=1, type=int,
                    help='Whether to transform the data into current source '
                         'density [1] or not [0].')
parser.add_argument('--sfreq', default=200, type=int,
                    help='Downsampling frequency.')
parser.add_argument('--things_eeg_2_dir',
                    default='/scratch/giffordale95/datasets/things_eeg_2',
                    type=str,
                    help='Directory of the THINGS EEG2 dataset.')
parser.add_argument('--project_dir', default=os.environ.get('EEG_FUSION_DATA', 'data'),
                    type=str,
                    help='Directory of the project folder.')
args = parser.parse_args()

print('>>> EEG data preprocessing <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220


# =============================================================================
# Epoch and sort the data
# =============================================================================
# After preprocessing, the EEG data is reshaped to:
# (Image conditions x EEG repetitions x EEG channels x EEG time points)
# This step is applied independently to the data of each partition and session.
epoched_test, _, ch_names, times = epoching(args, 'test', seed)
epoched_train, img_conditions_train, _, _ = epoching(args, 'training', seed)


# =============================================================================
# z-scorings
# =============================================================================
# z-scoring is applied independently to the data of each session.
zscored_test, zscored_train = zscore(args, epoched_test, epoched_train)
del epoched_test, epoched_train


# =============================================================================
# Merge and save the preprocessed data
# =============================================================================
# In this step the data of all sessions is merged into the shape:
# (Image conditions x EEG repetitions x EEG channels x EEG time points)
# Then, the preprocessed data of the test and training data partitions is saved.
save_prepr(args, zscored_test, zscored_train, img_conditions_train, ch_names,
           times, seed)
