from __future__ import annotations
import numpy as np


def _random_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a random orthogonal matrix via QR (the paper's Gram-Schmidt-based mask generation)."""
    G = rng.normal(size=(n, n))
    Q, R = np.linalg.qr(G)
    # Fix sign ambiguity of QR so masks are well-defined orthogonal matrices
    d = np.sign(np.diag(R))
    d[d == 0] = 1
    return Q * d


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


def run_pca_rom(party_data: dict[str, np.ndarray], y_aligned: np.ndarray, n_classes: int, r: int, seed: int = 7, gamma: float = 0.35):
    """
    Real privacy-preserving federated PCA under removable orthogonal masks (PCA-ROM), matching
    the VFedRS paper's Algorithm 1: parties never reveal raw features to each other; a semi-honest
    authority issues masks Q (sample-dim) and R (feature-dim) that cancel out exactly once removed.

    party_data: {'A': Xa_A, 'P1': Xa_1, ...}  each (n_aligned, dim_k), already centered per party.

    Note on `gamma`: the paper's supervised objective tr(V^T X^T L X V) with a one-hot label
    kernel L=YY^T is rank-deficient (rank <= n_classes-1), so it alone cannot supply more than
    n_classes-1 informative components. We regularize M = X^T L X + gamma * X^T X, blending the
    label-relevance term with the standard (unsupervised) covariance term — this keeps every
    component task-relevant while still spanning the full feature-variance subspace, and the
    regularizer passes through the exact same removable-mask algebra since X^T X masks/unmasks
    identically to X^T L X (both reduce to R^T(...)R once Q cancels).

    Returns federated representations H_fed for the aligned samples, the recovered principal
    components (per party), explained-variance info, and a numeric mask-correctness check
    (comparing against the unmasked, non-private computation) for the privacy report page.
    """
    rng = np.random.default_rng(seed)
    order = list(party_data.keys())
    dims = [party_data[k].shape[1] for k in order]
    n_aligned = next(iter(party_data.values())).shape[0]
    p_total = sum(dims)
    r = min(r, p_total, n_aligned)

    Xc_blocks = [party_data[k] - party_data[k].mean(axis=0, keepdims=True) for k in order]
    X_full = np.hstack(Xc_blocks)
    Y = one_hot(y_aligned, n_classes)
    Yc = Y - Y.mean(axis=0, keepdims=True)

    # --- Authority generates + distributes removable orthogonal masks ---
    Q = _random_orthogonal(n_aligned, rng)
    R = _random_orthogonal(p_total, rng)
    splits = np.cumsum(dims)[:-1]
    R_blocks = np.split(R, splits, axis=0)

    # --- Each party masks its own aligned data locally (raw features never leave the party) ---
    Xhat_blocks = [Q @ Xc_blocks[i] @ R_blocks[i] for i in range(len(order))]
    Xhat = sum(Xhat_blocks)
    Yhat = Q @ Yc

    # --- Active party computes the masked, regularized HSIC-supervised matrix M_hat = R^T M R ---
    M_hat = Xhat.T @ Yhat @ Yhat.T @ Xhat + gamma * (Xhat.T @ Xhat)
    eigvals_hat, eigvecs_hat = np.linalg.eigh(M_hat)
    ord_hat = np.argsort(eigvals_hat)[::-1]
    eigvals_hat = eigvals_hat[ord_hat]
    Vhat_r = eigvecs_hat[:, ord_hat][:, :r]

    # --- Unmask: since R is orthogonal, V = R @ V_hat exactly recovers the true PCs ---
    V_r = R @ Vhat_r
    V_blocks = np.split(V_r, splits, axis=0)

    H_fed = sum(Xc_blocks[i] @ V_blocks[i] for i in range(len(order)))

    # --- Ground-truth (non-private) computation, used ONLY to numerically verify the masks work ---
    M_true = X_full.T @ Yc @ Yc.T @ X_full + gamma * (X_full.T @ X_full)
    eigvals_t, eigvecs_t = np.linalg.eigh(M_true)
    ord_t = np.argsort(eigvals_t)[::-1]
    eigvals_true = eigvals_t[ord_t]
    V_true_r = eigvecs_t[:, ord_t][:, :r]
    H_true = X_full @ V_true_r

    # Subspace-invariant comparison (robust to sign flips / degenerate-eigenvalue rotations) of
    # the projection Gram matrices — the correct way to check "did unmasking recover the right
    # subspace", since individual eigenvectors are only unique up to rotation within tied eigenvalues.
    G_fed = H_fed @ H_fed.T
    G_true = H_true @ H_true.T
    denom = np.linalg.norm(G_true) + 1e-12
    mask_recon_error = float(np.linalg.norm(G_fed - G_true) / denom)

    total_var = float(np.sum(np.abs(eigvals_true))) + 1e-12
    explained = [float(abs(v)) / total_var for v in eigvals_hat[:r]]

    per_party_components = {k: V_blocks[i].tolist() for i, k in enumerate(order)}

    return {
        "H_fed": H_fed,
        "V_blocks": {k: V_blocks[i] for i, k in enumerate(order)},
        "explained_variance_ratio": explained,
        "mask_reconstruction_error": mask_recon_error,
        "n_components": r,
        "party_order": order,
        "party_dims": dict(zip(order, dims)),
        "per_party_components_preview": {k: np.array(v)[:5, : min(5, r)].round(4).tolist() for k, v in per_party_components.items()},
    }


def project_active_only(X_active_centered: np.ndarray, V_A: np.ndarray) -> np.ndarray:
    """What the active party can compute alone at inference time (no passive parties available),
    i.e. the vanilla-VFL / zero-padded baseline the paper evaluates against."""
    return X_active_centered @ V_A
