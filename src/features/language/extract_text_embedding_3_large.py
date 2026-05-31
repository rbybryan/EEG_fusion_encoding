"""Embed the GPT-4V image descriptions with OpenAI text-embedding-3-large.

This is the producer of the **headline language features** used by the fusion
encoding model. Each image was described by GPT-4V in five independent versions
(``v1`` .. ``v5``; see ``generate_gpt4v_captions.py`` and the cleaning step that
produces ``gpt4_5v_cleaned.csv``). Every version is embedded separately with
``text-embedding-3-large`` (3072-d), and the five embeddings are averaged to a
single per-image vector.

Input
-----
    <project_dir>/gpt4_features/gpt4_5v_cleaned.csv   columns: v1 .. v5

Output (a pickled dict saved with numpy)
----------------------------------------
    <project_dir>/gpt4_features/gpt4_responses_embedded_5v_large_cleaned.npy
        embedding_v1 .. embedding_v5 : list of (3072,) arrays (0 where empty)
        embedding_all               : list of (n_versions, 3072) arrays
        embedding_avg               : list of (3072,) arrays (mean over versions)
        dataframe                   : the input dataframe

The dimensionality-reduction (PCA) step applied before encoding is separate;
the encoding model consumes the PCA-reduced version of this file.

The OpenAI API key is read from the ``OPENAI_API_KEY`` environment variable.

Example
-------
    export OPENAI_API_KEY=...
    python extract_text_embedding_3_large.py --project_dir /path/to/encoding/data
"""

import argparse
import os
import os.path as op

import numpy as np
import pandas as pd
import tqdm
from openai import OpenAI

MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
N_VERSIONS = 5


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_dir', default='/scratch/byrong/encoding/data',
                        type=str, help='Project directory holding gpt4_features/.')
    parser.add_argument('--input_csv', default=None, type=str,
                        help='Cleaned 5-version descriptions CSV (default: '
                             '<project_dir>/gpt4_features/gpt4_5v_cleaned.csv).')
    parser.add_argument('--output_name',
                        default='gpt4_responses_embedded_5v_large_cleaned.npy',
                        type=str, help='Output file name (saved under '
                                       '<project_dir>/gpt4_features/).')
    return parser.parse_args()


def get_embedding(client, text, model=MODEL):
    """Return the embedding vector for a single string."""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)


def embed_versions(client, df):
    """Embed each of the five description versions for every image."""
    data_dict = {}
    for v in range(N_VERSIONS):
        col = 'v%d' % (v + 1)
        embeddings = []
        for i in tqdm.tqdm(range(len(df)), desc=col):
            text = df[col].iloc[i]
            if text != '':
                emb = get_embedding(client, text)
                if len(emb) != EMBED_DIM:
                    print(f'  row {i}: unexpected dim {len(emb)}')
                    embeddings.append(0)
                else:
                    embeddings.append(emb)
            else:
                embeddings.append(0)
        data_dict['embedding_v%d' % (v + 1)] = embeddings
    return data_dict


def average_versions(data_dict, df):
    """Average the per-version embeddings into one vector per image."""
    data_dict['embedding_all'] = []
    data_dict['embedding_avg'] = []
    for i in range(len(df)):
        if df['v1'].iloc[i] != '':
            stack = [data_dict['embedding_v%d' % (v + 1)][i]
                     for v in range(N_VERSIONS)
                     if isinstance(data_dict['embedding_v%d' % (v + 1)][i],
                                   np.ndarray)]
            if not stack:
                print(f'  row {i}: all versions empty')
            stacked = np.stack(stack)
            data_dict['embedding_all'].append(stacked)
            data_dict['embedding_avg'].append(np.average(stacked, axis=0))
        else:
            data_dict['embedding_all'].append(0)
            data_dict['embedding_avg'].append(0)
    return data_dict


def main():
    args = parse_args()
    if not os.environ.get('OPENAI_API_KEY'):
        raise SystemExit('Set the OPENAI_API_KEY environment variable.')
    client = OpenAI()  # reads OPENAI_API_KEY from the environment

    input_csv = args.input_csv or op.join(args.project_dir, 'gpt4_features',
                                          'gpt4_5v_cleaned.csv')
    df = pd.read_csv(input_csv)
    for v in range(N_VERSIONS):
        df['v%d' % (v + 1)] = df['v%d' % (v + 1)].fillna('')

    data_dict = embed_versions(client, df)
    data_dict['dataframe'] = df
    data_dict = average_versions(data_dict, df)

    out_path = op.join(args.project_dir, 'gpt4_features', args.output_name)
    os.makedirs(op.dirname(out_path), exist_ok=True)
    np.save(out_path, data_dict)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
