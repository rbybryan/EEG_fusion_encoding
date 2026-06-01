"""Channel-wise representational similarity analysis (RSA) between EEG and text.

For each subject, this script computes representational similarity between the
EEG responses and GPT-4 text-feature embeddings. Random samples of conditions
are drawn, representational dissimilarity is computed within a sliding time
window for every EEG channel, and the EEG dissimilarities are correlated with
the text-embedding dissimilarities using both Spearman and Pearson correlation.

Parameters
----------
text_embedding_file : str
    File name of the embedded GPT-4 text features.
project_dir : str
    Directory of the project data folder.
"""

import argparse
import os
import os.path as op
import random
import time

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity

# --- path configuration (portable defaults for public release) ---
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_DATA_ROOT = _os.environ.get('EEG_FUSION_DATA', _os.path.join(_REPO, 'data'))
# --- end path configuration ---


parser = argparse.ArgumentParser()
parser.add_argument('--text_embedding_file', type=str,
                    default='gpt4_features_embedded_large.npy',
                    help='File name of the embedded GPT-4 text features.')
parser.add_argument('--project_dir', default=_DATA_ROOT, type=str,
                    help='Directory of the project data folder.')
args = parser.parse_args()


### Loading the embeded text features ###
data_dict = np.load(
    op.join(args.project_dir, 'gpt4_features', args.text_embedding_file),
    allow_pickle=True).item()


train_index = data_dict["train_index"].copy()
text_train = data_dict['text_features_train_long'].copy()
del data_dict


def random_sample(all_sample_size, sample_size, seed=2024):
    """Split a shuffled set of conditions into equally sized chunks.

    Parameters
    ----------
    all_sample_size : int
        Total number of available conditions.
    sample_size : int
        Number of conditions per chunk. Note that this value is overridden
        internally to 100.
    seed : int, optional
        Seed for the random shuffle.

    Returns
    -------
    list of numpy.ndarray
        List of index chunks, each containing ``sample_size`` indices.
    """
    random.seed(seed)
    # Assuming you have a list of conditions/images
    all_conditions = np.arange(all_sample_size)  # assuming 10,000 conditions

    # Shuffle the list of conditions/images
    random.shuffle(all_conditions)

    # Number of images to sample each time
    sample_size = 100

    # Initialize an empty list to store the sampled images
    samples = []

    # Iterate over the shuffled list in chunks of sample_size
    for i in range(0, len(all_conditions), sample_size):
        # Get a chunk of sample_size images
        chunk = all_conditions[i:i + sample_size]
        # Add the chunk to the list of sampled images
        if len(chunk) == sample_size:
            samples.append(chunk)

    return samples


channels = 'preprocessed_data_all'
sample_size = 100
row_indices, col_indices = np.tril_indices(sample_size, -1)
t_win = 5
subject_corr = {}
for sub in range(1, 11):
    if sub != 5:

        tt = time.time()
        # load eeg data
        data_dir = os.path.join('eeg_dataset', channels, 'sub-' +
                                format(sub, '02'))
        training_file = 'preprocessed_eeg_training.npy'
        data = np.load(os.path.join(args.project_dir, data_dir, training_file),
                       allow_pickle=True).item()
        ch_names = data['ch_names']
        times = data['times']
        bio_train = data['preprocessed_eeg_data']
        bio_train = bio_train[train_index]
        # Averaging across repetitions
        y_train = np.mean(bio_train, 1)
        # Converting to float32 (for DNN training with Pytorch)
        y_train = np.float32(y_train)[:, :63, :]
        samples = random_sample(len(y_train), sample_size, sub)

        corr = {}
        corr['spearman_cos_eeg_cos_emb'] = np.zeros((len(samples), 63, 100 - t_win))
        corr['spearman_cor_eeg_cos_emb'] = np.zeros((len(samples), 63, 100 - t_win))
        corr['pearson_cos_eeg_cos_emb'] = np.zeros((len(samples), 63, 100 - t_win))
        corr['pearson_cor_eeg_cos_emb'] = np.zeros((len(samples), 63, 100 - t_win))

        for n, s in enumerate(samples):
            sampled_eeg = y_train[s]
            sampled_emb = text_train[s]
            cos_emb = cosine_similarity(sampled_emb)[row_indices, col_indices]
            for c in range(63):
                cos_eeg_tmp_ch, cor_eeg_tmp_ch = [], []
                for t, tp in enumerate(range(0, 100 - t_win)):
                    cos_eeg = cosine_similarity(
                        sampled_eeg[:, c, tp:tp + t_win])[row_indices, col_indices]
                    cor_eeg = np.corrcoef(
                        sampled_eeg[:, c, t:t + t_win])[row_indices, col_indices]
                    corr['spearman_cos_eeg_cos_emb'][n, c, t] = \
                        spearmanr(1 - cos_eeg, 1 - cos_emb)[0]
                    corr['spearman_cor_eeg_cos_emb'][n, c, t] = \
                        spearmanr(1 - cor_eeg, 1 - cos_emb)[0]
                    corr['pearson_cos_eeg_cos_emb'][n, c, t] = \
                        pearsonr(1 - cos_eeg, 1 - cos_emb)[0]
                    corr['pearson_cor_eeg_cos_emb'][n, c, t] = \
                        pearsonr(1 - cor_eeg, 1 - cos_emb)[0]

        print('subject ', sub, ' time elapsed: ', time.time() - tt, 's')
        subject_corr[sub] = corr

save_dir = os.path.join(args.project_dir, 'RSA')

if not op.exists(save_dir):
    os.makedirs(save_dir)

file_name = 'Channel_wise_%s_per_%d.npy' % (args.text_embedding_file, t_win)

np.save(os.path.join(save_dir, file_name), subject_corr)
