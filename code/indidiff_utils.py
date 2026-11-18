import numpy as np


def compute_icc_matrix(half1, half2):
    """
    Vectorised ICC(1,1) for every edge, given split-half connectivity matrices.

    With k=2, the unbiased within-subject variance is (h1 - h2)**2 / 2, and
    the between-subject mean square is k * Var_ddof1(subject means).

    Parameters
    ----------
    half1, half2 : (n_subjects, n_regions, n_regions) array-like

    Returns
    -------
    icc : (n_regions, n_regions) ndarray, NaN on the diagonal.
    """
    half1 = np.asarray(half1, dtype=np.float64)
    half2 = np.asarray(half2, dtype=np.float64)

    u = (half1 + half2) / 2
    w = (half1 - half2) ** 2 / 2

    msr = 2.0 * np.nanvar(u, axis=0, ddof=1)
    msw = np.nanmean(w, axis=0)

    denom = msr + msw
    icc = np.where(denom > 0, (msr - msw) / denom, 0.0)
    np.fill_diagonal(icc, np.nan)
    return icc


def _center_norm_rows(X):
    """Mean-centre and unit-norm each row, so row dot products give Pearson r."""
    X = np.asarray(X, dtype=np.float64)
    X = X - np.nanmean(X, axis=1, keepdims=True)
    X = np.nan_to_num(X, nan=0.0)  # NaN edges contribute 0 after centering
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


def compute_identifiability_matrix(half1_flat, half2_flat):
    """
    Pearson correlation between every pair of (subject-i half-1, subject-j half-2).

    Parameters
    ----------
    half1_flat, half2_flat : (n_subjects, n_edges) array-like

    Returns
    -------
    identi : (n_subjects, n_subjects) ndarray, identi[i, j] = corr(half1[i], half2[j]).
    """
    h1 = _center_norm_rows(half1_flat)
    h2 = _center_norm_rows(half2_flat)
    return h1 @ h2.T


def identifiability_stats(identi_matrix):
    """
    Extract Iself, Iother, Idiff from an identifiability matrix.

    Iother for subject i is the mean of all off-diagonal values in row i and
    column i combined, following the Finn et al. convention.
    """
    n = identi_matrix.shape[0]
    iself = np.diag(identi_matrix).copy()

    iother = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        row_off = identi_matrix[i, mask]
        col_off = identi_matrix[mask, i]
        iother[i] = np.mean(np.concatenate([row_off, col_off]))

    idiff = iself - iother
    return iself, iother, idiff


def permutation_null_idiff(identi_matrix, n_perm=1000, seed=0):
    """
    Null distribution of Iself, Iother, Idiff under shuffled subject labels.

    For each permutation p (a derangement of subject indices), the entry
    identi[i, p[i]] plays the role of Iself for subject i, and Iother is
    the mean of all off-diagonal cells in row i and column p[i] of the
    original matrix, excluding identi[i, i] and the new "self" cell.

    Returns
    -------
    iself_null, iother_null, idiff_null : each (n_perm, n_subjects) ndarray
    """
    rng = np.random.default_rng(seed)
    identi = np.asarray(identi_matrix, dtype=np.float64)
    n = identi.shape[0]
    idx = np.arange(n)

    iself_null = np.empty((n_perm, n), dtype=np.float64)
    iother_null = np.empty((n_perm, n), dtype=np.float64)

    for k in range(n_perm):
        while True:
            p = rng.permutation(n)
            if not np.any(p == idx):
                break
        for i in range(n):
            pi = p[i]
            iself_null[k, i] = identi[i, pi]
            row_mask = idx != pi
            col_mask = idx != i
            row_vals = identi[i, row_mask]
            col_vals = identi[col_mask, pi]
            iother_null[k, i] = np.concatenate([row_vals, col_vals]).mean()

    idiff_null = iself_null - iother_null
    return iself_null, iother_null, idiff_null
