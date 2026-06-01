"""Build GloVe (non-contextual) sentence embeddings of the GPT-4V image
descriptions, as a SIMPLER language baseline for the fusion encoding model
(Reviewer 2, Point 7).

Rationale
---------
R2.7 asks whether the LLM advantage reflects merely lexical content or richer
(contextual) semantic structure. GloVe gives a word-level, non-contextual
sentence embedding (mean of word vectors). If the LLM (text-embedding-3-large)
continues to outperform GloVe as the language component of the fusion model,
the effect depends on contextual/relational structure, not lexical content alone.

We replicate the exact image ordering and train_index used to build the other
language embeddings (extract_embeddings.py), so the output aligns 1:1 with the
EEG and vision features in the encoding pipeline.

Output (consumed by encoding_model.py via --language_model glove):
    /scratch/byrong/encoding/data/gpt4_features/gpt4_features_embedded_by_glove_pca.npy
    keys: text_features_train_long (Ntrain, 300), text_features_test_long (200, 300), train_index
"""
import os
import os.path as op
import re

import numpy as np
import pandas as pd

PROJECT_DIR = '/scratch/byrong/encoding/data'
GLOVE_NAME = 'glove-wiki-gigaword-300'      # classic GloVe, 6B tokens, 300-d
OUT = op.join(PROJECT_DIR, 'gpt4_features', 'gpt4_features_embedded_by_glove_pca.npy')
TOKEN_RE = re.compile(r"[a-z]+")


def collect_responses():
    """Replicate extract_embeddings.py: walk sorted images, gather v1..v5,
    record train_index (images with non-empty v1)."""
    df = pd.read_csv(op.join(PROJECT_DIR, 'gpt4_features', 'gpt4_5v_cleaned.csv')).fillna('')
    vmap = {row['image']: [row[f'v{k}'] for k in range(1, 6)] for _, row in df.iterrows()}

    def sorted_jpgs(subdir):
        imgs = []
        for root, _, files in os.walk(op.join(PROJECT_DIR, 'image_set', subdir)):
            for f in files:
                if f.endswith('.jpg'):
                    imgs.append(op.join(root, f))
        imgs.sort()
        return imgs

    train_imgs = sorted_jpgs('training_images')
    test_imgs = sorted_jpgs('test_images')

    train_index, resp_train = [], [[] for _ in range(5)]
    for i, p in enumerate(train_imgs):
        name = p.split('/')[-1]
        vs = vmap.get(name, [''] * 5)
        if vs[0] != '':
            train_index.append(i)
            for k in range(5):
                resp_train[k].append(vs[k])

    resp_test = [[] for _ in range(5)]
    for p in test_imgs:
        name = p.split('/')[-1]
        vs = vmap.get(name, [''] * 5)
        if vs[0] != '':
            for k in range(5):
                resp_test[k].append(vs[k])

    return train_index, resp_train, resp_test


def embed_list(texts, kv):
    dim = kv.vector_size
    out = np.zeros((len(texts), dim), dtype=np.float32)
    n_oov_docs = 0
    for i, t in enumerate(texts):
        toks = TOKEN_RE.findall(str(t).lower())
        vecs = [kv[w] for w in toks if w in kv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
        else:
            n_oov_docs += 1
    return out, n_oov_docs


def main():
    print('Collecting GPT-4V descriptions ...')
    train_index, resp_train, resp_test = collect_responses()
    n_train = len(resp_train[0])
    n_test = len(resp_test[0])
    print(f'train descriptions/version: {n_train} | test: {n_test} | train_index: {len(train_index)}')

    # sanity vs canonical train_index
    can = np.asarray(list(np.load(op.join(
        PROJECT_DIR, 'gpt4_features',
        'gpt4_features_embedded_5v_large_cleaned_avg_pca.npy'),
        allow_pickle=True).item()['train_index']))
    match = (len(can) == len(train_index)) and bool(np.array_equal(can, np.asarray(train_index)))
    print(f'train_index matches canonical TEL file: {match}')
    if not match:
        print('  WARNING: train_index mismatch — using locally-derived index but verify alignment.')

    print(f'Loading GloVe vectors ({GLOVE_NAME}) ...')
    import gensim.downloader as api
    kv = api.load(GLOVE_NAME)
    print(f'GloVe loaded: vocab={len(kv.index_to_key)}, dim={kv.vector_size}')

    emb_train = np.zeros((5, n_train, kv.vector_size), dtype=np.float32)
    emb_test = np.zeros((5, n_test, kv.vector_size), dtype=np.float32)
    for k in range(5):
        emb_train[k], oov_tr = embed_list(resp_train[k], kv)
        emb_test[k], oov_te = embed_list(resp_test[k], kv)
        print(f'  v{k+1}: train OOV-empty docs={oov_tr}, test OOV-empty docs={oov_te}')

    # average over the 5 versions (matches the *_avg representation used for TEL)
    text_train = emb_train.mean(axis=0)     # (n_train, 300)
    text_test = emb_test.mean(axis=0)       # (200, 300)
    print(f'avg train {text_train.shape}, avg test {text_test.shape}')

    data_dict = {
        'text_features_train_long': text_train,
        'text_features_test_long': text_test,
        'train_index': list(train_index),
    }
    np.save(OUT, data_dict)
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
