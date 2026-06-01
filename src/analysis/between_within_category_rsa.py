"""Between- vs within-category RSA: does the DNN-LLM representational overlap
survive removal of coarse (object-)category structure? (Reviewer 1, Point 2)

Logic
-----
The manuscript reports r = 0.24 between the pairwise cosine-similarity RDMs of
the vision DNN (CORnet-S) and the LLM (text-embedding-3-large) over the full
training set (Fig. 1B). A reviewer asks whether this overlap merely reflects
shared coarse category structure (same object -> similar in both spaces) or
persists as finer-grained shared structure.

We therefore split image pairs by THINGS object category (1,654 concepts,
10 exemplars each):
  * BETWEEN-category pairs : the two images are different objects. The DNN-LLM
    correlation here is dominated by coarse category structure.
  * WITHIN-category  pairs : the two images are the SAME object (e.g. two
    different "dog" images). Category identity is held constant, so only
    fine-grained, instance-level differences remain -- no categorical structure.

If the DNN-LLM correlation remains positive WITHIN category, the overlap is not
reducible to category labels. Comparing TRAINED vs UNTRAINED CORnet then asks
whether any residual within-category alignment reflects *learned* structure
(convergent learning) or pre-existing architectural bias.

Pipeline is matched to the manuscript headline number (notebook
similarity-between-model.ipynb): cosine_similarity on the first 1000 features
of each representation, Pearson r over the lower-triangle of the RDMs.

Outputs
-------
analysis/between_within_category_rsa_summary.csv
analysis/between_within_category_rsa_arrays.npz
rebuttal/figures/between_within_category_rsa.png
"""

import os
import os.path as op
import time

import numpy as np
from scipy.stats import linregress, pearsonr
from sklearn.metrics.pairwise import cosine_similarity

# --- path configuration (portable defaults for public release) ---
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_DATA_ROOT = _os.environ.get('EEG_FUSION_DATA', _os.path.join(_REPO, 'data'))
# --- end path configuration ---


PROJECT_DIR = _DATA_ROOT
REPO_DIR = _REPO
LLM_FILE = 'gpt4_features_embedded_5v_large_cleaned_avg_pca.npy'  # text-embedding-3-large, 5v avg
# Fig 1B representation: the CORnet 3000-PC features (first 1000 reproduce the
# manuscript's r=0.24), NOT the kernel-PCA encoding features (which give 0.08).
PCA_ROOTS = [_os.path.join(_DATA_ROOT, 'pca_feature_maps')]
N_FEAT = 1000          # first-1000 of the 3000 PCs, matching Fig 1B (r=0.2411)
N_EXEMPLARS = 10       # THINGS-EEG2 training: 1654 concepts x 10 exemplars
N_BOOT = 1000
SEED = 20200220


def load_llm():
    """Load the LLM text-embedding features and the training-image index.

    Returns
    -------
    text : numpy.ndarray
        Float64 array of shape ``(n_train, 3072)`` with the full
        text-embedding-3-large features for the valid training images.
    train_index : list of int
        Indices (into the 16,540 THINGS-EEG2 training images) of the images
        that survive GPT-4V validity filtering.
    """
    d = np.load(op.join(PROJECT_DIR, 'gpt4_features', LLM_FILE), allow_pickle=True).item()
    text = np.asarray(d['text_features_train_long'])          # (16495, 3072) — use FULL 3072 (Fig 1B)
    train_index = list(d['train_index'])                      # 16495 indices into 16540
    return text.astype(np.float64), train_index


def load_cornet_all_layers(pretrained):
    """Load the concatenated all-layers CORnet-S PCA feature maps.

    Parameters
    ----------
    pretrained : str
        ``'True'`` for the ImageNet-trained network or ``'False'`` for the
        randomly initialised (untrained) network.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(n_images, 1000)`` with the all-layers PCA features.

    Raises
    ------
    FileNotFoundError
        If no matching feature file is found under ``PCA_ROOTS``.
    """
    for root in PCA_ROOTS:
        p = op.join(root, 'cornet_s', f'pretrained-{pretrained}',
                    'layers-all', 'pca_feature_maps_training.npy')
        if op.isfile(p):
            al = np.load(p, allow_pickle=True).item()['all_layers']   # (16540, 1000)
            return np.asarray(al)
    raise FileNotFoundError(f'CORnet all_layers (pretrained-{pretrained}) not found')


def rdm_lower(feats):
    """Lower-triangle vector of the cosine-similarity RDM (float32 to save RAM)."""
    sim = cosine_similarity(feats).astype(np.float32)
    iu = np.tril_indices(sim.shape[0], -1)
    vec = sim[iu]
    return vec, iu


def main():
    """Run the between- vs within-category RSA and write the summary outputs."""
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    llm, train_index = load_llm()
    n_img = llm.shape[0]
    print(f'LLM features: {llm.shape}; train images: {n_img}')

    # Per-image THINGS concept id (10-exemplar blocks), then restrict to the
    # 16,495 GPT-4V-valid images via train_index.
    concept_full = np.arange(16540) // N_EXEMPLARS
    concept = concept_full[np.asarray(train_index)]
    assert concept.shape[0] == n_img, (concept.shape, n_img)
    print(f'n concepts (restricted): {len(np.unique(concept))}')

    # LLM RDM lower triangle (shared across DNN conditions)
    print('Computing LLM RDM ...')
    y_full, iu = rdm_lower(llm)
    del llm
    row_c = concept[iu[0]]
    col_c = concept[iu[1]]
    within = (row_c == col_c)                 # same THINGS concept
    print(f'total pairs: {within.size:,} | within-category: {within.sum():,} '
          f'| between-category: {(~within).sum():,}')

    # Pre-group within-category pair indices by concept for a concept-level bootstrap
    within_idx = np.flatnonzero(within)
    within_concept = row_c[within_idx]
    order = np.argsort(within_concept, kind='stable')
    within_idx_sorted = within_idx[order]
    wc_sorted = within_concept[order]
    uniq_c, starts = np.unique(wc_sorted, return_index=True)
    ends = np.r_[starts[1:], len(wc_sorted)]
    concept_to_pairidx = {int(c): within_idx_sorted[s:e]
                          for c, s, e in zip(uniq_c, starts, ends)}

    results = {}
    arrays = {'within_mask_sum': int(within.sum()),
              'between_mask_sum': int((~within).sum())}

    for label, pretrained in [('trained', 'True'), ('untrained', 'False')]:
        print(f'\n=== CORnet-S ({label}) ===')
        al_full = load_cornet_all_layers(pretrained)
        # The Fig 1B feature file is already the 16,495 valid images; older files are 16,540
        al_full = al_full[np.asarray(train_index)] if al_full.shape[0] == 16540 else al_full
        dnn = al_full[:, :N_FEAT].astype(np.float64)
        x_full, iu2 = rdm_lower(dnn)
        del dnn
        assert np.array_equal(iu2[0][:3], iu[0][:3])  # same indexing

        def pear(mask=None):
            if mask is None:
                r = pearsonr(x_full, y_full)[0]
                s = linregress(x_full, y_full).slope
            else:
                r = pearsonr(x_full[mask], y_full[mask])[0]
                s = linregress(x_full[mask], y_full[mask]).slope
            return float(r), float(s)

        r_all, s_all = pear(None)
        r_win, s_win = pear(within)
        r_bet, s_bet = pear(~within)
        print(f'  overall : r = {r_all:.4f}  slope = {s_all:.4f}')
        print(f'  between : r = {r_bet:.4f}  slope = {s_bet:.4f}')
        print(f'  within  : r = {r_win:.4f}  slope = {s_win:.4f}')

        # Concept-level bootstrap of the WITHIN-category r (the load-bearing claim)
        boot = np.empty(N_BOOT)
        all_concepts = np.array(list(concept_to_pairidx.keys()))
        for b in range(N_BOOT):
            samp = rng.choice(all_concepts, size=all_concepts.size, replace=True)
            idx = np.concatenate([concept_to_pairidx[int(c)] for c in samp])
            boot[b] = pearsonr(x_full[idx], y_full[idx])[0]
        ci = np.percentile(boot, [2.5, 97.5])
        print(f'  within  bootstrap 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]')

        results[label] = dict(r_all=r_all, slope_all=s_all,
                              r_between=r_bet, slope_between=s_bet,
                              r_within=r_win, slope_within=s_win,
                              within_ci_low=float(ci[0]), within_ci_high=float(ci[1]))
        arrays[f'{label}_within_boot'] = boot
        del x_full

    # Trained vs untrained within-category difference (paired bootstrap reuse)
    diff = arrays['trained_within_boot'] - arrays['untrained_within_boot']
    dci = np.percentile(diff, [2.5, 97.5])
    p_two = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    print(f'\nWithin-category r difference (trained - untrained): '
          f'{diff.mean():.4f} 95% CI [{dci[0]:.4f}, {dci[1]:.4f}] p~{p_two:.4f}')
    results['within_trained_minus_untrained'] = dict(
        mean=float(diff.mean()), ci_low=float(dci[0]), ci_high=float(dci[1]),
        p_boot=float(p_two))

    # ---- save ----
    os.makedirs(op.join(REPO_DIR, 'analysis'), exist_ok=True)
    import csv
    csv_path = op.join(REPO_DIR, 'analysis', 'between_within_category_rsa_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['condition', 'metric', 'value'])
        for label in ['trained', 'untrained']:
            for k, v in results[label].items():
                w.writerow([label, k, v])
        for k, v in results['within_trained_minus_untrained'].items():
            w.writerow(['within_trained_minus_untrained', k, v])
    np.savez(op.join(REPO_DIR, 'analysis', 'between_within_category_rsa_arrays.npz'),
             **arrays)
    print(f'\nSaved {csv_path}')

    # ---- figure ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    groups = ['Overall', 'Between-\ncategory', 'Within-\ncategory']
    tr = [results['trained'][k] for k in ['r_all', 'r_between', 'r_within']]
    un = [results['untrained'][k] for k in ['r_all', 'r_between', 'r_within']]
    tr_err = [[0, 0], [0, 0],
              [tr[2] - results['trained']['within_ci_low'],
               results['trained']['within_ci_high'] - tr[2]]]
    un_err = [[0, 0], [0, 0],
              [un[2] - results['untrained']['within_ci_low'],
               results['untrained']['within_ci_high'] - un[2]]]
    tr_err = np.array(tr_err).T
    un_err = np.array(un_err).T
    x = np.arange(3)
    w = 0.38
    fig, ax = plt.subplots(figsize=(4.2, 3.2), dpi=200)
    ax.bar(x - w / 2, tr, w, yerr=tr_err, capsize=3, label='Trained CORnet-S',
           color='#1f77b4')
    ax.bar(x + w / 2, un, w, yerr=un_err, capsize=3, label='Untrained CORnet-S',
           color='#aec7e8')
    for xi, (a, b) in enumerate(zip(tr, un)):
        ax.text(xi - w / 2, a + 0.005, f'{a:.3f}', ha='center', va='bottom', fontsize=7)
        ax.text(xi + w / 2, b + 0.005, f'{b:.3f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel('DNN–LLM RDM correlation (Pearson r)')
    ax.set_title('DNN–LLM representational overlap by category structure')
    ax.axhline(0, color='k', lw=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    os.makedirs(op.join(REPO_DIR, 'rebuttal', 'figures'), exist_ok=True)
    figp = op.join(REPO_DIR, 'rebuttal', 'figures', 'between_within_category_rsa.png')
    fig.savefig(figp, bbox_inches='tight')
    print(f'Saved {figp}')
    print(f'\nTotal time: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
