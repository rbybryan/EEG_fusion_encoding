"""Cell-for-cell equivalence check: batched kernels vs the per-cell oracle.

Confirms that ``grid_search_batched`` / ``grid_search_fusion_batched`` reproduce
the reference per-cell ``Grid_search`` / ``Grid_search_fusion`` (predictions AND
selected hyper-parameters), so the published numbers are unchanged when using the
fast path. Run:

    python src/encoding/validate_fast.py
"""

import numpy as np

from utils import (
    Grid_search,
    Grid_search_fusion,
    grid_search_batched,
    grid_search_fusion_batched,
)


def make_data(n_train=140, n_test=40, p0=8, p1=6, T=12, seed=0):
    """Synthetic data with heterogeneous dependence on both blocks plus noise."""
    rng = np.random.default_rng(seed)
    X0_tr = rng.standard_normal((n_train, p0)).astype(np.float32)
    X1_tr = rng.standard_normal((n_train, p1)).astype(np.float32)
    X0_te = rng.standard_normal((n_test, p0)).astype(np.float32)
    X1_te = rng.standard_normal((n_test, p1)).astype(np.float32)
    # Targets depend on both blocks with target-varying SNR, so different cells
    # prefer different alpha / weight.
    Wt0 = rng.standard_normal((p0, T))
    Wt1 = rng.standard_normal((p1, T))
    scale = np.linspace(0.2, 3.0, T)  # vary SNR across targets
    Y_tr = (X0_tr @ Wt0 + X1_tr @ Wt1) * scale + 0.7 * rng.standard_normal((n_train, T))
    Y_tr = Y_tr.astype(np.float32)
    return X0_tr, X1_tr, X0_te, X1_te, Y_tr


def test_single_space():
    """Batched single-space search must match the per-cell oracle."""
    X0_tr, X1_tr, X0_te, X1_te, Y_tr = make_data()
    X_tr, X_te = X0_tr, X0_te
    alpha_range = np.logspace(-3, 7, 25)
    T = Y_tr.shape[1]

    Yp_b, a_b, fl_b = grid_search_batched(X_tr, X_te, Y_tr, alpha_range, kfold=5)

    Yp_o = np.zeros((X_te.shape[0], T))
    a_o = np.zeros(T)
    for t in range(T):
        Yp_o[:, t], a_o[t], _ = Grid_search(
            X_tr, X_te, Y_tr[:, t], None, alpha_range=alpha_range, metric='r2', kfold=5)

    dpred = np.abs(Yp_b - Yp_o).max()
    n_alpha_mismatch = int((a_b != a_o).sum())
    print(f"[single]  max|dpred|={dpred:.2e}   alpha mismatches={n_alpha_mismatch}/{T}")
    assert dpred < 1e-4, dpred
    assert n_alpha_mismatch == 0
    return True


def test_fusion():
    """Batched fusion search must match the per-cell oracle."""
    X0_tr, X1_tr, X0_te, X1_te, Y_tr = make_data()
    alpha_range = np.logspace(-3, 7, 20)
    weight_list = np.linspace(0, 1000, 11) / 1000.0  # 0.0 .. 1.0
    T = Y_tr.shape[1]

    Yp_b, a_b, w_b, fl_b = grid_search_fusion_batched(
        X0_tr, X0_te, X1_tr, X1_te, Y_tr, alpha_range, weight_list, kfold=5)

    Yp_o = np.zeros((X0_te.shape[0], T))
    a_o = np.zeros(T)
    w_o = np.zeros(T)
    for t in range(T):
        Yp_o[:, t], a_o[t], w_o[t], _ = Grid_search_fusion(
            X0_tr, X0_te, X1_tr, X1_te, Y_tr[:, t], Y_tr[:, t],
            alpha_range=alpha_range, weight_list=weight_list, metric='r2', kfold=5)

    dpred = np.abs(Yp_b - Yp_o).max()
    n_alpha_mismatch = int((a_b != a_o).sum())
    n_w_mismatch = int((w_b != w_o).sum())
    print(f"[fusion]  max|dpred|={dpred:.2e}   alpha mismatches={n_alpha_mismatch}/{T}"
          f"   weight mismatches={n_w_mismatch}/{T}")
    assert dpred < 1e-4, dpred
    assert n_alpha_mismatch == 0
    assert n_w_mismatch == 0
    return True


if __name__ == '__main__':
    ok_s = test_single_space()
    ok_f = test_fusion()
    print("\nALL EQUIVALENCE CHECKS PASSED" if (ok_s and ok_f) else "FAILED")
