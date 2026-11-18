import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import itertools
import networkx as nx
import igraph as ig
import leidenalg as la
import bct
import cpnet
from infomap import Infomap
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.utils import check_random_state
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.linalg import eigvalsh, solve
from msapy import msa, utils as msa_ut
from numba import njit, prange
import numba


# #######################################################
# PREPROC
#########################################################


def prop_thresh(A, prop=0.1):
    """Keep only the top prop proportion of weights, set others to zero."""
    n = A.shape[0]
    triu_indices = np.triu_indices(n, k=1)
    weights = A[triu_indices]
    threshold = np.percentile(weights, 100 * (1 - prop))
    A_thresh = np.where(A >= threshold, A, 0)
    return A_thresh

def is_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)

def make_symmetric(data):
    data = (data + data.T) / 2
    return data

def get_length(data):
    """Weight-to-length remap: -log(w / (max(w) * (1 + eps))).

    Natural log, with a small margin on the denominator so max-weight edges
    map to a small positive length instead of zero. Zero weights map to +inf.
    """
    max_val = np.nanmax(data)
    if not np.isfinite(max_val) or max_val <= 0:
        return np.full_like(data, np.inf, dtype=float)
    with np.errstate(divide='ignore'):
        return -np.log(data / (max_val * (1.0 + 1e-6)))

def rank_normalise(W):
    """Map edge weights to their percentile rank in [0, 1].
    i.e the empirical probability integral transform (PIT)"""
    from scipy.stats import rankdata
    flat = W.flatten()
    ranked = rankdata(flat, method='average') / len(flat)
    return ranked.reshape(W.shape).astype(np.float32)


def minmax_normalisation(W):
    W = (W - np.nanmin(W)) / (np.nanmax(W) - np.nanmin(W))
    return W

def log_minmax_normalisation(data, fill_diagonal=True):
    data = np.nan_to_num(np.log(data), neginf=0, posinf=0)
    data = (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))
    if fill_diagonal:
        np.fill_diagonal(data, 0.0)
    return data

def strength_normalisation(adjacency_matrix):
    """
    Subside the effect of high-strength (or high-degree) nodes.
    https://royalsocietypublishing.org/doi/full/10.1098/rsif.2008.0484

    Note: input must be non-negative. Negative row sums would yield
    NaN/complex values from power(s, -0.5).
    """
    strength: np.ndarray = adjacency_matrix.sum(1)
    with np.errstate(divide='ignore'):
        normalized_strength: np.ndarray = np.power(strength, -0.5)
    normalized_strength[~np.isfinite(normalized_strength)] = 0.0
    diagonalized_normalized_strength: np.ndarray = np.diag(normalized_strength)
    normalized_adjacency_matrix: np.ndarray = (
        diagonalized_normalized_strength @ adjacency_matrix @ diagonalized_normalized_strength)
    return normalized_adjacency_matrix

def matrix_to_igraph(M, directed=False):
    if directed:
        return ig.Graph.Weighted_Adjacency(M.tolist(), 
                                            mode=ig.ADJ_DIRECTED, 
                                            attr="weight")
    else:
        return ig.Graph.Weighted_Adjacency(M.tolist(), 
                                            mode=ig.ADJ_UNDIRECTED, 
                                            attr="weight")


# #######################################################
# TOPOLOGY
#########################################################

@njit(parallel=True, cache=True, fastmath=True)
def _local_efficiency_jit(W):
    """
    Weighted local efficiency (Rubinov & Sporns 2010; BCT `efficiency_wei`).

    Formula
    -------
    E_loc(i) = 1 / (k_i * (k_i - 1))
               * Σ_{j≠h ∈ N_i}  w_ij^{1/3} · w_ih^{1/3} · e_jh(N_i)

    where e_jh(N_i) = 1 / d_jh is the inverse shortest-path length between
    neighbours j and h in the subgraph induced on N_i (node i excluded),
    with edge lengths taken on the cube-rooted weights: d_direct = 1 / w^{1/3}.
    """
    n = W.shape[0]
    E_loc = np.zeros(n)
    W_cb = W ** (1.0 / 3.0)

    for i in prange(n):
        nbrs = np.where(W[i] > 0)[0]
        k = len(nbrs)
        if k < 2:
            continue

        # distances on the cube-rooted neighbourhood subgraph
        D = np.empty((k, k))
        for a in range(k):
            D[a, a] = 0.0
            for b in range(a + 1, k):
                wcb_ab = W_cb[nbrs[a], nbrs[b]]
                d = 1.0 / wcb_ab if wcb_ab > 0.0 else np.inf
                D[a, b] = d
                D[b, a] = d

        for mid in range(k):
            for a in range(k):
                d_a_mid = D[a, mid]
                if d_a_mid == np.inf:
                    continue
                for b in range(k):
                    through = d_a_mid + D[mid, b]
                    if through < D[a, b]:
                        D[a, b] = through

        eff_sum = 0.0
        for a in range(k):
            j = nbrs[a]
            wcb_ij = W_cb[i, j]
            for b in range(k):
                if a == b:
                    continue
                d_jh = D[a, b]
                if d_jh == np.inf:
                    continue
                h = nbrs[b]
                eff_sum += wcb_ij * W_cb[i, h] / d_jh

        E_loc[i] = eff_sum / (k * (k - 1))

    return E_loc


def comp_efficiency(W, local=False, n_threads=1):
    """
    Global or local efficiency of a graph. Undirected graphs only.
        - global: average inverse shortest path length between all pairs of nodes.
        - local: per-node efficiency (scalar array, shape N), BCT formula.

    n_threads controls Numba parallelism for local efficiency. Default 1 is
    correct when called from within a joblib parallel pool; increase for
    standalone use.
    """
    if local:
        numba.set_num_threads(n_threads)
        return _local_efficiency_jit(W.astype(np.float64))
    return bct.efficiency_wei(W, local=False)

def comp_assortativity(W, flag=0):
    """
    Degree assortativity of a graph. Directed and undirected (see flag).
        - Pearson correlation between the degrees of connected nodes.
        - positive assortativity = hubs tend to connect to hubs,
          negative assortativity = hubs tend to connect to non-hubs.
    flag: 0 for undirected, 1 for out-strength/-in-strength correlation
    """
    if W.size == 0 or not np.any(W):
        return 0.0
    with np.errstate(invalid='ignore'):
        result = bct.assortativity_wei(W, flag=flag)
    return result if np.isfinite(result) else 0.0

def comp_partC(W, ci, degree='undirected'):
    """
    Diversity of intermodular connections of each node. Directed and undirected (see degree).

    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        binary/weighted directed/undirected connection matrix
    ci : Nx1 :obj:`numpy.ndarray`
        community affiliation vector
    degree : {'undirected', 'in', 'out'}, optional
        Flag to describe nature of graph. 'undirected': For undirected graphs,
        'in': Uses the in-degree, 'out': Uses the out-degree

    Returns
    -------
    P : Nx1 :obj:`numpy.ndarray`
        participation coefficient
    """
    
    if degree == 'in':
        W = W.T

    _, ci = np.unique(ci, return_inverse=True)
    ci += 1

    n = len(W)  # number of vertices
    Ko = np.sum(W, axis=1)  # (out) degree
    Gc = np.dot((W != 0), np.diag(ci))  # neighbor community affiliation
    Kc2 = np.zeros((n,))  # community-specific neighbors

    for i in range(1, int(np.max(ci)) + 1):
        Kc2 += np.square(np.sum(W * (Gc == i), axis=1))

    P = np.ones((n,)) - Kc2 / np.square(Ko)
    # P=0 if for nodes with no (out) neighbors
    P[np.where(np.logical_not(Ko))] = 0

    return P

def comp_mod(W, ci, directed=False):
    """
    Modularity of a given partition of a graph. Directed and undirected (see directed).

    Parameters
    ----------
    W : NxN np.ndarray  — weighted connection matrix
    ci : Nx1 np.ndarray — community affiliation vector
    directed : bool     — use directed formula (bct.modularity_dir) when True,
                          undirected (bct.modularity_und) when False
    """
    if directed:
        _, Q = bct.modularity_dir(W, kci=ci.astype(float))
    else:
        _, Q = bct.modularity_und(W, kci=ci.astype(float))
    return Q

def spec_radius(M):
    """Spectral radius (largest eigenvalue magnitude). Directed and undirected."""
    if not isinstance(M, (csr_matrix)):
        M_sparse = csr_matrix(M).asfptype()
    else:
        M_sparse = M.asfptype()
    eigenvalues = eigs(M_sparse, k=1, which='LM', return_eigenvectors=False)
    return np.abs(eigenvalues[0])

def laplacian(M):
    """Graph Laplacian: L = D - A. Undirected graphs only (symmetric input assumed)."""
    M = np.asarray(M, dtype=np.float64)
    L = -M.copy()
    np.fill_diagonal(L, M.sum(axis=1))
    return L

def compute_von_ent(M):
    """
    Graph Von Neumann entropy. Undirected graphs only.
        - treating normalized Laplacian as a density operator
        - large entropy = more complex, less redundant structure.
    """

    L = laplacian(M)
    trace_L = np.trace(L)
    if trace_L == 0:
        return 0.0
    
    evals = eigvalsh(L)
    rho_evals = evals / trace_L # normalize to sum to 1 (density matrix rho)

    nz_evals = rho_evals[rho_evals > 1e-12]
    vne = -np.sum(nz_evals * np.log(nz_evals)) # plogp hehe!

    return vne

def compute_kirchhoff_index(M):
    """
    Graph Kirchhoff index. Undirected graphs only.
        - sum of effective resistances between all pairs of nodes.
        - small index = more efficient information flow.
    """
    n = M.shape[0]
    L = laplacian(M)
    tL = np.trace(L)
    if tL == 0:
        return 0.0
    evals = eigvalsh(L)
    nz = (evals / tL)[evals / tL > 1e-12]
    if len(nz) == 0:
        return 0.0
    return n / tL * np.sum(1.0 / nz)

def compute_spectral_gap(M):
    """
    Graph Spectral Gap. Undirected graphs only.
        - difference between the two largest eigenvalues of the matrix.
        - large gap = fast information expansion/mixing.
    """
    evals = eigvalsh(M)
    if len(evals) < 2:
        return 0.0
    return evals[-1] - evals[-2]

def comp_fv(W):
    """
    Algebraic connectivity (Fiedler value) of a graph. Undirected graphs only.
        - second smallest eigenvalue of the Laplacian matrix.
        - large value = more robust to node/edge removal, less bottlenecks.
    """
    L = laplacian(W)
    return np.sort(np.linalg.eigvalsh(L))[1]

def comp_s_metric(W, directed=False):
    """S-metric: sum of strength(u)*strength(v) over all edges. Directed and undirected (see directed)."""
    if directed:
        out_strength = W.sum(axis=1)
        in_strength = W.sum(axis=0)
        i, j = np.where(W > 0)
        return float((out_strength[i] * in_strength[j]).sum())
    strength = W.sum(axis=1)
    i, j = np.triu_indices(len(strength), k=1)
    mask = W[i, j] > 0
    return float((strength[i[mask]] * strength[j[mask]]).sum())

def build_shapley_space(n, n_permutations, seed=42):
    """Pre-compute the MSA permutation/combination/complement spaces.
    These depend only on n, n_permutations, and seed — not on the data.
    """
    nodes = list(range(n))
    node_perms = msa.make_permutation_space(elements=nodes, n_permutations=n_permutations, random_seed=seed)
    node_combs = msa.make_combination_space(permutation_space=node_perms)
    node_compl = msa.make_complement_space(combination_space=node_combs, elements=nodes)
    return {'node_perms': node_perms, 'node_combs_template': node_combs, 'node_compl_template': node_compl}


def multi_perturb_shapely(M, func,
                          seed=42,
                          n_permutations=1000,
                          verbose=False,
                          n_jobs=1,
                          pre_comp={'node_perms': None,
                                   'node_combs_template': None,
                                   'node_compl_template': None}
                          ):
    """
    Multi-perturbation Shapley value analysis

    Parameters
    ----------
    M : np.ndarray
        Adjacency matrix of the graph.
        Want it to be connected and undirected!
    func : function
        A function that takes a graph (or its adjacency matrix) 
        and returns a scalar value representing some property of the graph 
        (e.g., global efficiency).
    
    """

    n = M.shape[0]
    nodes = list(range(n))

    def game(complements, matrix):
        nc = len(complements)
        if nc == 0:
            return np.float64(func(matrix))
        elif nc == n:
            return np.float64(0.0)
        mask = np.ones(n, dtype=bool)
        mask[list(complements)] = False
        sub = matrix[np.ix_(mask, mask)]
        return np.float64(func(sub))

    if pre_comp['node_perms'] is not None and pre_comp['node_combs_template'] is not None and pre_comp['node_compl_template'] is not None:
        node_perms = pre_comp['node_perms']
        node_combs_template = pre_comp['node_combs_template']
        node_compl_template = pre_comp['node_compl_template']
    else:
        node_perms = msa.make_permutation_space(elements=nodes,
                                                n_permutations=n_permutations,
                                                random_seed=seed)
        node_combs_template = msa.make_combination_space(permutation_space=node_perms)
        node_compl_template = msa.make_complement_space(combination_space=node_combs_template,
                                                        elements=nodes)

    if verbose:
        print(np.shape(node_perms))
        print(f'Nodes: {nodes}\n')
        print(f'Permuted nodes: {node_perms[0]}')
        print(f'Number of lesion combinations: {len(node_combs_template)}')

    eff, _ = msa_ut.parallelized_take_contributions(complement_space=node_compl_template,
                                                  combination_space=node_combs_template,
                                                  objective_function=game,
                                                  objective_function_params={'matrix': M},
                                                  n_cores=n_jobs)

    eff_shapley = msa.get_shapley_table(contributions=eff, permutation_space=node_perms)
    eff_shapley = msa_ut.sorter(eff_shapley)

    intact_eff = func(M)
    d_eff = msa_ut.distribution_of_processing(shapley_vector=eff_shapley.mean())

    # Shapley efficiency axiom: Σ phi_i = v(N) - v(o) = intact_eff - 0.
    # MC-sampled Shapley has nonzero variance, so use a relative tolerance.
    shap_sum = eff_shapley.mean().sum()
    tol = max(0.05, 0.05 * abs(intact_eff))
    if abs(intact_eff - shap_sum) > tol:
        raise ValueError(
            f"Shapley values don't sum to intact - lesioned "
            f"(intact={intact_eff}, sum={shap_sum}, tol={tol}). "
            f"Check game function and contributions."
        )
    
    return eff_shapley.sort_index(axis=1).to_numpy().T, d_eff


def multi_perturb_shapely_joint(M, n_permutations=50, seed=42, n_jobs=1, pre_comp=None):
    n = M.shape[0]

    if pre_comp is not None and pre_comp.get('node_perms') is not None:
        node_perms = pre_comp['node_perms']
    else:
        nodes = list(range(n))
        node_perms = msa.make_permutation_space(
            elements=nodes, n_permutations=n_permutations, random_seed=seed)

    if hasattr(node_perms, 'values'):
        perm_list = node_perms.values.tolist()
    else:
        perm_list = [list(p) for p in node_perms]

    def _process_perm(perm):
        vne_vals = np.zeros(n, dtype=np.float64)
        kf_vals = np.zeros(n, dtype=np.float64)
        idx = []
        prev_vne, prev_kf = 0.0, 0.0

        for node in perm:
            idx.append(node)
            sub = M[np.ix_(idx, idx)]
            sub_n = len(idx)
            L = laplacian(sub)
            tL = np.trace(L)
            evals = eigvalsh(L)

            if tL == 0:
                cur_vne = 0.0
                cur_kf = 0.0
            else:
                rho = evals / tL
                nz = rho[rho > 1e-12]
                cur_vne = float(-np.sum(nz * np.log(nz))) if len(nz) > 0 else 0.0
                cur_kf = float(sub_n / tL * np.sum(1.0 / nz)) if len(nz) > 0 else 0.0

            vne_vals[node] = cur_vne - prev_vne
            kf_vals[node] = cur_kf - prev_kf
            prev_vne, prev_kf = cur_vne, cur_kf

        return vne_vals, kf_vals

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_perm)(perm) for perm in perm_list
    )

    vne_shapley = np.stack([r[0] for r in results], axis=1)
    kf_shapley = np.stack([r[1] for r in results], axis=1)

    vne_intact = float(compute_von_ent(M))
    kf_intact = float(compute_kirchhoff_index(M))
    for name, arr, intact in [
        ('von_neumann_entropy', vne_shapley, vne_intact),
        ('kirchhoff_index', kf_shapley, kf_intact),
    ]:
        shap_sum = arr.mean(axis=1).sum()
        tol = max(0.05, 0.05 * abs(intact))
        if abs(intact - shap_sum) > tol:
            raise ValueError(
                f"Joint Shapley axiom failed for {name}: "
                f"intact={intact:.4f}, sum={shap_sum:.4f}, tol={tol:.4f}"
            )

    return {'von_neumann_entropy': vne_shapley, 'kirchhoff_index': kf_shapley}


def multi_perturb_shapely_all_measures(M, n_permutations=50, seed=42, n_jobs=1, pre_comp=None):
    """
    Single complement-space pass computing Shapley values for assortativity,
    VNE, and Kirchhoff simultaneously. Each coalition is evaluated once:
    one submatrix extraction, one eigvalsh, one assortativity call.
    """
    n = M.shape[0]
    nodes = list(range(n))

    _pre = pre_comp or {}

    if _pre.get('node_perms') is not None:
        node_perms = _pre['node_perms']
    else:
        node_perms = msa.make_permutation_space(elements=nodes, n_permutations=n_permutations, random_seed=seed)

    if _pre.get('node_combs_template') is not None:
        node_combs = _pre['node_combs_template']
    else:
        node_combs = msa.make_combination_space(permutation_space=node_perms)

    if _pre.get('node_compl_template') is not None:
        node_compl = _pre['node_compl_template']
    else:
        node_compl = msa.make_complement_space(combination_space=node_combs, elements=nodes)

    def _triple_eval(complements, matrix):
        nc = len(complements)
        if nc == n:
            return (0.0, 0.0, 0.0)
        if nc == 0:
            sub = matrix
        else:
            mask = np.ones(n, dtype=bool)
            mask[list(complements)] = False
            sub = matrix[np.ix_(mask, mask)]
        sub_n = sub.shape[0]
        if sub.size == 0 or not np.any(sub):
            assort = 0.0
        else:
            with np.errstate(invalid='ignore'):
                r = bct.assortativity_wei(sub, 0)
            assort = float(r) if np.isfinite(r) else 0.0
        L = laplacian(sub)
        tL = np.trace(L)
        if tL == 0:
            return (assort, 0.0, 0.0)
        evals = eigvalsh(L)
        rho = evals / tL
        nz = rho[rho > 1e-12]
        if len(nz) == 0:
            return (assort, 0.0, 0.0)
        vne = float(-np.sum(nz * np.log(nz)))
        kf = float(sub_n / tL * np.sum(1.0 / nz))
        return (assort, vne, kf)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_triple_eval)(complement, M) for complement in node_compl
    )

    assort_contribs = dict(zip(node_combs, [r[0] for r in results]))
    vne_contribs = dict(zip(node_combs, [r[1] for r in results]))
    kf_contribs = dict(zip(node_combs, [r[2] for r in results]))

    assort_shapley = msa.get_shapley_table(contributions=assort_contribs, permutation_space=node_perms)
    vne_shapley = msa.get_shapley_table(contributions=vne_contribs, permutation_space=node_perms)
    kf_shapley = msa.get_shapley_table(contributions=kf_contribs, permutation_space=node_perms)

    assort_arr = assort_shapley.sort_index(axis=1).to_numpy().T
    vne_arr = vne_shapley.sort_index(axis=1).to_numpy().T
    kf_arr = kf_shapley.sort_index(axis=1).to_numpy().T

    assort_intact = float(comp_assortativity(M))
    vne_intact = float(compute_von_ent(M))
    kf_intact = float(compute_kirchhoff_index(M))
    for name, arr, intact in [
        ('assortativity', assort_arr, assort_intact),
        ('von_neumann_entropy', vne_arr, vne_intact),
        ('kirchhoff_index', kf_arr, kf_intact),
    ]:
        shap_sum = arr.mean(axis=1).sum()
        tol = max(0.05, 0.05 * abs(intact))
        if abs(intact - shap_sum) > tol:
            raise ValueError(
                f"Shapley axiom failed for {name}: "
                f"intact={intact:.4f}, sum={shap_sum:.4f}, tol={tol:.4f}"
            )

    return {'assortativity': assort_arr, 'von_neumann_entropy': vne_arr, 'kirchhoff_index': kf_arr}


def multi_perturb_shapely_joint_complement(M, n_permutations=50, seed=42, n_jobs=1, pre_comp=None):
    n = M.shape[0]
    nodes = list(range(n))

    _pre = pre_comp or {}

    if _pre.get('node_perms') is not None:
        node_perms = _pre['node_perms']
    else:
        node_perms = msa.make_permutation_space(elements=nodes, n_permutations=n_permutations, random_seed=seed)

    if _pre.get('node_combs_template') is not None:
        node_combs = _pre['node_combs_template']
    else:
        node_combs = msa.make_combination_space(permutation_space=node_perms)

    if _pre.get('node_compl_template') is not None:
        node_compl = _pre['node_compl_template']
    else:
        node_compl = msa.make_complement_space(combination_space=node_combs, elements=nodes)

    def _joint_eval(complements, matrix):
        nc = len(complements)
        if nc == n:
            return (0.0, 0.0)
        if nc == 0:
            sub = matrix
        else:
            mask = np.ones(n, dtype=bool)
            mask[list(complements)] = False
            sub = matrix[np.ix_(mask, mask)]
        sub_n = sub.shape[0]
        L = laplacian(sub)
        tL = np.trace(L)
        if tL == 0:
            return (0.0, 0.0)
        evals = eigvalsh(L)
        rho = evals / tL
        nz = rho[rho > 1e-12]
        if len(nz) == 0:
            return (0.0, 0.0)
        vne = float(-np.sum(nz * np.log(nz)))
        kf = float(sub_n / tL * np.sum(1.0 / nz))
        return (vne, kf)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_joint_eval)(complement, M) for complement in node_compl
    )

    vne_contribs = dict(zip(node_combs, [r[0] for r in results]))
    kf_contribs = dict(zip(node_combs, [r[1] for r in results]))

    vne_shapley = msa.get_shapley_table(contributions=vne_contribs, permutation_space=node_perms)
    kf_shapley = msa.get_shapley_table(contributions=kf_contribs, permutation_space=node_perms)

    vne_arr = vne_shapley.sort_index(axis=1).to_numpy().T
    kf_arr = kf_shapley.sort_index(axis=1).to_numpy().T

    vne_intact = float(compute_von_ent(M))
    kf_intact = float(compute_kirchhoff_index(M))
    for name, arr, intact in [
        ('von_neumann_entropy', vne_arr, vne_intact),
        ('kirchhoff_index', kf_arr, kf_intact),
    ]:
        shap_sum = arr.mean(axis=1).sum()
        tol = max(0.05, 0.05 * abs(intact))
        if abs(intact - shap_sum) > tol:
            raise ValueError(
                f"Joint Shapley axiom failed for {name}: "
                f"intact={intact:.4f}, sum={shap_sum:.4f}, tol={tol:.4f}"
            )

    return {'von_neumann_entropy': vne_arr, 'kirchhoff_index': kf_arr}


#########################################################
# COMMUNITY DETECTION
#########################################################


def _leiden_partition(G, r):
    return np.array(la.find_partition(
        G, la.CPMVertexPartition,
        weights='weight', resolution_parameter=r
    ).membership, dtype=np.int32)

def leiden_cpm(W, r=0.5, directed=False):
    """Leiden CPM community detection. Directed and undirected (see directed)."""
    return _leiden_partition(matrix_to_igraph(W, directed=directed), r)


def merge_small_communities(ci, consensus, min_size=5):
    """
    Merge communities smaller than min_size into their highest-affinity neighbour.

    Affinity between two communities is the mean co-assignment value in the
    consensus matrix across all cross-community node pairs. Small communities
    are preferentially merged with other small communities first; only when no
    other small community exists does a small community merge into a larger one.
    The smallest community is processed first each iteration. Labels are
    relabelled 0..K-1 on return.

    Parameters
    ----------
    ci : np.ndarray (N,)
        Community labels from consensus clustering.
    consensus : np.ndarray (N, N)
        Co-assignment matrix produced during consensus clustering (values in [0, 1]).
    min_size : int, optional
        Minimum number of nodes per community. Default 5.

    Returns
    -------
    ci_merged : np.ndarray (N,) int
        Merged and relabelled community labels.
    """
    ci = ci.copy().astype(int)
    while True:
        labels, counts = np.unique(ci, return_counts=True)
        if counts.min() >= min_size:
            break

        small_mask = counts < min_size
        idx = np.argmin(counts)
        small_label = labels[idx]
        small_nodes = np.where(ci == small_label)[0]

        # prefer merging with other small communities; fall back to large ones
        other_small = labels[small_mask & (labels != small_label)]
        candidates = other_small if len(other_small) > 0 else labels[~small_mask]

        best_label = candidates[0]
        best_affinity = -1.0
        for target in candidates:
            target_nodes = np.where(ci == target)[0]
            affinity = float(consensus[np.ix_(small_nodes, target_nodes)].mean())
            if affinity > best_affinity:
                best_affinity = affinity
                best_label = target
        ci[small_nodes] = best_label

    _, ci = np.unique(ci, return_inverse=True)
    return ci


def _multires_consensus_matrix(matrix,
                               rmin=None, rmax=None,
                               rnum_1=50, rnum_2=50,
                               space="linear",
                               n_jobs=1):
    """
    Build the consensus co-assignment matrix from a multiresolution Leiden CPM sweep.

    Two-pass procedure: first sweep across [rmin, rmax] with rnum_1 resolutions to
    identify the range of resolutions that yield non-trivial partitions, then a
    second denser sweep with rnum_2 resolutions across that range. The consensus
    matrix is the mean co-assignment across the second-pass partitions.

    Returns
    -------
    consensus : (N, N) np.float32
        Co-assignment matrix, values in [0, 1].
    """
    assert space in {"linear", "geometric"}

    if rmin is None:
        rmin = np.nanmin(matrix)
    if rmax is None:
        rmax = np.nanmax(matrix)

    directed = not is_symmetric(matrix)
    G = matrix_to_igraph(matrix, directed=directed)
    N = G.vcount()

    if space == "linear":
        resolutions = np.linspace(rmin, rmax, num=rnum_1)
    else:
        resolutions = np.geomspace(10**-6, rmax, num=rnum_1)

    all_memberships = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_leiden_partition)(G, r) for r in resolutions
    )

    partition_rs = set()
    seen_partitions = set()
    for r, membership in zip(resolutions, all_memberships):
        t = tuple(membership)
        unq_count = len(np.unique(membership))
        if 1 < unq_count < N:
            if t not in seen_partitions:
                partition_rs.add(r)
                seen_partitions.add(t)

    mn = min(partition_rs)
    mx = max(partition_rs)

    if space == "linear":
        resolutions = np.linspace(mn, mx, num=rnum_2)
    else:
        resolutions = np.geomspace(10**-6, mx, num=rnum_2)

    memberships = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_leiden_partition)(G, r) for r in resolutions
    )

    M = np.array(memberships)
    consensus = np.mean(M[:, :, None] == M[:, None, :], axis=0).astype(np.float32)
    return consensus


def multires_consensus_comdet(matrix,
                              rmin=None, rmax=None,
                              rnum_1=50, rnum_2=50,
                              n_iterations=2,
                              space="linear",
                              consensus_method="modularity",
                              resolution_parameter=0.5,
                              min_community_size=1,
                              return_consensus=False,
                              n_jobs=1):
    """
    Multiresolution consensus community detection. Directed and undirected (auto-detected from matrix symmetry).
    https://arxiv.org/abs/1710.02249

    Builds the consensus matrix via _multires_consensus_matrix, then partitions it
    with Leiden (modularity or CPM) and optionally merges small communities.

    Parameters
    ----------
    matrix : (N x N) array_like
        Weighted connection matrix.
    rmin, rmax : float, optional
        Resolution sweep bounds (defaults: matrix min/max).
    rnum_1, rnum_2 : int
        Number of resolutions in the first and second sweeps.
    n_iterations : int, optional
        Leiden n_iterations for the consensus clustering step. Default 2.
    space : str, optional
        "linear" (default) or "geometric" resolution sampling.
    consensus_method : str, optional
        "modularity" (default): ModularityVertexPartition.
        "cpm": CPMVertexPartition with resolution_parameter.
    resolution_parameter : float, optional
        Resolution for CPMVertexPartition when consensus_method="cpm". Default 0.5.
    min_community_size : int, optional
        Communities smaller than this are merged into their highest-affinity
        neighbour. Default 1 (no merging).
    return_consensus : bool, optional
        If True, return (comms, consensus_matrix). Default False.

    Returns
    -------
    comms : np.ndarray
        Community assignment of each node (zero-indexed).
    consensus : np.ndarray (N, N), only if return_consensus=True
    """
    assert consensus_method in {"modularity", "cpm"}

    consensus = _multires_consensus_matrix(
        matrix, rmin=rmin, rmax=rmax,
        rnum_1=rnum_1, rnum_2=rnum_2,
        space=space, n_jobs=n_jobs,
    )
    N = consensus.shape[0]

    G_consensus = matrix_to_igraph(consensus, directed=False)
    if consensus_method == "modularity":
        partition = la.find_partition(G_consensus,
                                      la.ModularityVertexPartition,
                                      weights="weight",
                                      n_iterations=n_iterations)
    else:
        partition = la.find_partition(G_consensus,
                                      la.CPMVertexPartition,
                                      resolution_parameter=resolution_parameter,
                                      weights="weight",
                                      n_iterations=n_iterations)
    comms = np.zeros(N, dtype=int)
    for i, comm in enumerate(partition):
        comms[list(comm)] = i

    if min_community_size > 1:
        comms = merge_small_communities(comms, consensus, min_size=min_community_size)

    if return_consensus:
        return comms, consensus
    return comms


def multires__consensus_hierarchical_comdet(matrix,
                                 rmin=None, rmax=None,
                                 rnum_1=50, rnum_2=50,
                                 space="linear",
                                 K_range=range(2, 51),
                                 linkage_method='average',
                                 criterion='silhouette',
                                 return_consensus=False,
                                 return_K_results=False,
                                 n_jobs=1):
    """
    Multiresolution consensus + agglomerative hierarchical consolidation with
    silhouette-based K selection. Undirected graphs only.

    Follows the consensus-clustering framework of Lancichinetti & Fortunato (2012),
    with hierarchical consolidation in the spirit of Yeo et al. (2011).

    Parameters
    ----------
    matrix : (N x N) array_like
        Weighted connection matrix.
    rmin, rmax, rnum_1, rnum_2, space : see multires_consensus_comdet.
    K_range : iterable of int
        Range of K values to test. Default range(2, 51).
    linkage_method : str
        Linkage criterion. Default 'average'.
    criterion : {'silhouette', 'modularity', 'manual'}
        K-selection criterion. Default 'silhouette' (Yeo et al. 2011 style).
        See hierarchical_K_from_consensus for details.
    return_consensus : bool
        If True, include the consensus matrix in the return.
    return_K_results : bool
        If True, include the full K-sweep dict.

    Returns
    -------
    comms : np.ndarray
        Community assignment at K* (zero-indexed).
    Followed by, in order if requested: consensus matrix, K_results dict.
    """
    consensus = _multires_consensus_matrix(
        matrix, rmin=rmin, rmax=rmax,
        rnum_1=rnum_1, rnum_2=rnum_2,
        space=space, n_jobs=n_jobs,
    )
    K_results = hierarchical_K_from_consensus(
        consensus, K_range=K_range,
        linkage_method=linkage_method, criterion=criterion,
    )
    comms = K_results['partitions'][K_results['K_optimal']]

    out = (comms,)
    if return_consensus:
        out = out + (consensus,)
    if return_K_results:
        out = out + (K_results,)
    if len(out) == 1:
        return out[0]
    return out


def hierarchical_K_from_consensus(consensus, K_range=range(2, 51),
                                  linkage_method='average',
                                  criterion='silhouette'):
    """
    Hierarchical clustering on a precomputed consensus matrix with K selection.

    Builds an agglomerative dendrogram on (1 - consensus), cuts at every K in K_range,
    scores each cut by the chosen criterion, and returns the K that maximises the score.

    Parameters
    ----------
    consensus : (N, N) np.ndarray
        Consensus co-assignment matrix (values in [0, 1]).
    K_range : iterable of int, optional
        Range of K values to test. Default range(2, 51).
    linkage_method : str, optional
        Linkage criterion: 'average', 'complete', 'single', 'ward'. Default 'average'.
    criterion : {'silhouette', 'modularity', 'manual'}
        Score used to select K*.
            'silhouette' (default): mean silhouette score using (1 - consensus) as
                distance. Doesn't suffer from the modularity resolution limit;
                tends to favour finer partitions when they're internally cohesive.
                Standard in brain mapping (e.g. Yeo et al. 2011).
            'modularity': Newman-Girvan Q on the consensus matrix. Subject to the
                modularity resolution limit (Fortunato & Barthelemy 2007), so on
                large graphs it often picks K=2.
            'manual': K_optimal is the median of K_range; caller picks K manually.

    Returns
    -------
    out : dict with keys
        K_range : np.ndarray of K values tried
        scores : np.ndarray of scores at each K (NaN if criterion is 'manual')
        K_optimal : int, K that maximises the score
        partitions : dict mapping K -> (N,) int array of community labels
        criterion : str, the criterion used
    """
    import warnings
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    assert criterion in {'silhouette', 'modularity', 'manual'}

    dist = np.clip(1.0 - consensus.astype(np.float64), 0.0, None)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method=linkage_method)

    K_arr = np.asarray(list(K_range), dtype=int)
    partitions = {}
    scores = np.full(len(K_arr), np.nan)

    if criterion == 'silhouette':
        from sklearn.metrics import silhouette_score

    for i, K in enumerate(K_arr):
        labels = fcluster(Z, t=K, criterion='maxclust') - 1
        partitions[int(K)] = labels.astype(int)
        n_unique = len(np.unique(labels))
        if n_unique < 2 or n_unique >= len(labels):
            continue
        if criterion == 'modularity':
            try:
                scores[i] = comp_mod(consensus.astype(np.float64), labels.astype(int))
            except Exception:
                scores[i] = np.nan
        elif criterion == 'silhouette':
            try:
                scores[i] = silhouette_score(dist, labels, metric='precomputed')
            except Exception:
                scores[i] = np.nan

    all_nan = (criterion != 'manual') and np.all(np.isnan(scores))
    if criterion == 'manual' or all_nan:
        K_optimal = int(K_arr[len(K_arr) // 2])
        if all_nan:
            warnings.warn(
                f'All {criterion} scores were NaN (degenerate dendrogram or single-cluster cuts); '
                f'falling back to K={K_optimal} (median of K_range). '
                'Consensus matrix likely lacks separable structure.'
            )
    else:
        K_optimal = int(K_arr[np.nanargmax(scores)])

    return {
        'K_range': K_arr,
        'scores': scores,
        'K_optimal': K_optimal,
        'partitions': partitions,
        'criterion': criterion,
    }


def walktrap_comdet(matrix, steps=4):
    """
    Community detection using Walktrap algorithm. Undirected graphs only.

    Parameters:
    -----------
    matrix : np.ndarray
        Adjacency matrix (weighted or unweighted)
    steps : int, default=4
        Length of random walks (typically 3-8)
        - Lower values (3-4): More, smaller communities
        - Higher values (5-8): Fewer, larger communities

    Returns:
    --------
    comms : np.ndarray
        Community assignments (same format as your infomap function)
    """
    
    directed = not is_symmetric(matrix)
    G = matrix_to_igraph(matrix, directed=directed)

    dendrogram = G.community_walktrap(weights='weight', steps=steps)
    clustering = dendrogram.as_clustering()
    comms = np.array(clustering.membership, dtype=int)
    
    return comms

def infomap_comdet(matrix,
                   directed=False,
                   markov_time=1.0,
                   variable_markov_time=True,
                   two_level=True,
                   trials=100,
                   option='infomap'):
    """Infomap community detection. Directed and undirected (see directed)."""
    if option == 'infomap':
        flow_model = "directed" if directed else "undirected"
        args = [
            f"--flow-model {flow_model}",
            f"--markov-time {markov_time}",
            f"--num-trials {trials}",
            "--silent"
        ]
        
        if two_level:
            args.append("--two-level")
        if variable_markov_time:
            args.append("--variable-markov-time")
            
        im = Infomap(" ".join(args))
        
        rows, cols = np.where(matrix > 0)
        for r, c in zip(rows, cols):
            w = float(matrix[r, c])
            im.add_link(int(r), int(c), w)
            
        im.run()
        
        comms = np.zeros(matrix.shape[0], dtype=int)
        for node in im.tree:
            if node.is_leaf:
                p_id = node.physicalId 
                m_id = node.moduleId()
                comms[p_id] = m_id - 1

    else:
        G_ig = matrix_to_igraph(matrix, directed=directed)
        comms = G_ig.community_infomap(edge_weights='weight', trials=trials).membership
    
    return comms


#########################################################
# CORE-PERIPHERY
#########################################################

def _to_graph(A):
    return nx.from_numpy_array(A)


def _dict_to_array(d, n, dtype=float):
    if dtype == float:
        return np.array([d[i] if d[i] is not None else np.nan for i in range(n)])
    return np.array([d[i] if d[i] is not None else -1 for i in range(n)], dtype=int)


def _fit_km_config(G, num_runs):
    algo = cpnet.KM_config(num_runs=num_runs)
    algo.detect(G)
    return algo, algo.get_pair_id(), algo.get_coreness()


def _fit_minres(G, num_runs, n_jobs):
    algo = cpnet.MINRES(num_runs=num_runs)
    algo.n_jobs = n_jobs
    algo.detect(G)
    return algo, algo.get_pair_id(), algo.get_coreness()


def _cp_qstest(c_dict, x_dict, G, algo, significance_level=0.05, num_of_rand_net=100):
    sig_c, sig_x, significant, p_values = cpnet.qstest(
        c_dict, x_dict, G, algo,
        significance_level=significance_level,
        num_of_rand_net=num_of_rand_net,
    )
    for i, (sig, p) in enumerate(zip(significant, p_values)):
        print(f'  pair {i}: p={p:.4f}  significant={sig}', flush=True)
    return sig_c, sig_x, p_values


def detect_cp_continuous(A, num_runs=10, n_jobs=1,
                         significance_test=False,
                         significance_level=0.05, num_of_rand_net=100):
    """
    MINRES — single core-periphery structure, continuous coreness in [0, 1].

    Parameters
    ----------
    A : ndarray (n, n)
        Symmetric weighted adjacency matrix, non-negative.
    num_runs : int
        Random restarts for the ADAM gradient descent.
    n_jobs : int
        Parallel jobs across restarts (set as instance attribute on MINRES).
    significance_test : bool
        If True, run qstest and include sig_coreness and p_values in output.
    significance_level : float
        Alpha for qstest (only used when significance_test=True).
    num_of_rand_net : int
        Surrogate networks for qstest (only used when significance_test=True).

    Returns
    -------
    dict with keys:
        coreness      : ndarray (n,) float, continuous coreness
        sig_coreness  : ndarray (n,) float, NaN for non-significant (if tested)
        p_values      : list[float] (if tested)
    """
    n = A.shape[0]
    G = _to_graph(A)
    algo, c_dict, x_dict = _fit_minres(G, num_runs, n_jobs)
    result = {'coreness': _dict_to_array(x_dict, n, dtype=float)}
    if significance_test:
        sig_c, sig_x, p_values = _cp_qstest(
            c_dict, x_dict, G, algo, significance_level, num_of_rand_net)
        result['sig_coreness'] = _dict_to_array(sig_x, n, dtype=float)
        result['p_values'] = p_values
    return result


def detect_cp_discrete(A, num_runs=50,
                       significance_test=False,
                       significance_level=0.05, num_of_rand_net=100):
    """
    KM_config — multiple core-periphery pairs, discrete coreness (0/1).

    Parameters
    ----------
    A : ndarray (n, n)
        Symmetric weighted adjacency matrix, non-negative.
    num_runs : int
        Random restarts for the label-switching optimiser.
    significance_test : bool
        If True, run qstest and include sig_* and p_values in output.
    significance_level : float
        Alpha for qstest (only used when significance_test=True).
    num_of_rand_net : int
        Surrogate networks for qstest (only used when significance_test=True).

    Returns
    -------
    dict with keys:
        pair_id       : ndarray (n,) int, CP pair index per node
        coreness      : ndarray (n,) int, 1=core / 0=periphery
        sig_pair_id   : ndarray (n,) int, -1 for non-significant pairs (if tested)
        sig_coreness  : ndarray (n,) float, NaN for non-significant pairs (if tested)
        p_values      : list[float] (if tested)
    """
    n = A.shape[0]
    G = _to_graph(A)
    algo, c_dict, x_dict = _fit_km_config(G, num_runs)
    n_pairs = len(set(c_dict.values()))
    print(f'KM_config: {n_pairs} CP pair(s)', flush=True)
    result = {
        'pair_id':  _dict_to_array(c_dict, n, dtype=int),
        'coreness': _dict_to_array(x_dict, n, dtype=int),
    }
    if significance_test:
        sig_c, sig_x, p_values = _cp_qstest(
            c_dict, x_dict, G, algo, significance_level, num_of_rand_net)
        result['sig_pair_id']  = _dict_to_array(sig_c, n, dtype=int)
        result['sig_coreness'] = _dict_to_array(sig_x, n, dtype=float)
        result['p_values'] = p_values
    return result


#########################################################
# NULLS
#########################################################

def strength_preserving_rand_sa_undir(
    A,
    rewiring_iter=10,
    nstage=100,
    niter=10000,
    temp=1000,
    frac=0.5,
    full_rand=False,
    energy_type="sse",
    energy_func=None,
    R=None,
    connected=None,
    verbose=False,
    seed=None,
):
    """
    Strength-preserving network randomization using simulated annealing.

    Randomize an undirected weighted network, while preserving
    the degree and strength sequences using simulated annealing.

    This function allows for a flexible choice of energy function.

    Parameters
    ----------
    A : (N, N) array-like
        Undirected weighted connectivity matrix
    rewiring_iter : int, optional
        Rewiring parameter. Default = 10.
        Each edge is rewired approximately rewiring_iter times.
    nstage : int, optional
        Number of annealing stages. Default = 100.
    niter : int, optional
        Number of iterations per stage. Default = 10000.
    temp : float, optional
        Initial temperature. Default = 1000.
    frac : float, optional
        Fractional decrease in temperature per stage. Default = 0.5.
    energy_type: str, optional
        Energy function to minimize. Can be either:
            'sse': Sum of squared errors between strength sequence vectors
                   of the original network and the randomized network
            'max': Maximum absolute error
            'mae': Mean absolute error
            'mse': Mean squared error
            'rmse': Root mean squared error
        Default = 'sse'.
    energy_func: callable, optional
        Callable with two positional arguments corresponding to
        two strength sequence numpy arrays that returns an energy value.
        Overwrites “energy_type”.
        See “energy_type” for specifying a predefined energy type instead.
    R : (N, N) array-like, optional
        Pre-randomized connectivity matrix.
        If None, a rewired connectivity matrix is generated using the
        Maslov & Sneppen algorithm.
        Default = None.
    connected: bool, optional
        Whether to ensure connectedness of the randomized network.
        By default, this is inferred from data.
    verbose: bool, optional
        Whether to print status to screen at the end of every stage.
        Default = False.
    seed: float, optional
        Random seed. Default = None.

    full_rand: bool, optional
        Whether to perform a full randomization instead of Maslov&Sneppen rewiring.
        Only use when have very dense networks to save time!
        Default = False.

    Returns
    -------
    B : (N, N) array-like
        Randomized connectivity matrix
    min_energy : float
        Minimum energy obtained by annealing

    Notes
    -----
    Uses Maslov & Sneppen rewiring model to produce a
    surrogate connectivity matrix, B, with the same
    size, density, and degree sequence as A.
    The weights are then permuted to optimize the
    match between the strength sequences of A and B
    using simulated annealing.

    This function is adapted from a function written in MATLAB
    by Richard Betzel.

    References
    ----------
    Misic, B. et al. (2015) Cooperative and Competitive Spreading Dynamics
    on the Human Connectome. Neuron.
    Milisav, F. et al. (2024) A simulated annealing algorithm for
    randomizing weighted networks.
    """
    try:
        A = np.asarray(A)
    except TypeError as err:
        msg = "A must be array_like. Received: {}.".format(type(A))
        raise TypeError(msg) from err

    if frac > 1 or frac <= 0:
        msg = "frac must be between 0 and 1. " "Received: {}.".format(frac)
        raise ValueError(msg)

    rs = check_random_state(seed)

    n = A.shape[0]
    s = np.sum(A, axis=1)  # strengths of A

    # Maslov & Sneppen rewiring
    if R is None:
        if full_rand:
            u, v = np.tril(A, k=-1).nonzero()
            wts = np.tril(A, k=-1)[(u, v)]
            np.random.shuffle(wts)
            B = np.zeros_like(A)
            B[(u, v)] = wts
            B = B + B.T
        else:
            # ensuring connectedness if the original network is connected
            if connected is None:
                connected = False if bct.number_of_components(A) > 1 else True
            if connected:
                B = bct.randmio_und_connected(A, rewiring_iter, seed=seed)[0]
            else:
                B = bct.randmio_und(A, rewiring_iter, seed=seed)[0]
    else:
        B = R.copy()

    u, v = np.triu(B, k=1).nonzero()  # upper triangle indices
    wts = np.triu(B, k=1)[(u, v)]  # upper triangle values
    m = len(wts)
    sb = np.sum(B, axis=1)  # strengths of B

    if energy_func is not None:
        energy = energy_func(s, sb)
    elif energy_type == "sse":
        energy = np.sum((s - sb) ** 2)
    elif energy_type == "max":
        energy = np.max(np.abs(s - sb))
    elif energy_type == "mae":
        energy = np.mean(np.abs(s - sb))
    elif energy_type == "mse":
        energy = np.mean((s - sb) ** 2)
    elif energy_type == "rmse":
        energy = np.sqrt(np.mean((s - sb) ** 2))
    else:
        msg = (
            "energy_type must be one of 'sse', 'max', "
            "'mae', 'mse', or 'rmse'. Received: {}.".format(energy_type)
        )
        raise ValueError(msg)

    energymin = energy
    wtsmin = wts.copy()

    if verbose:
        print("\ninitial energy {:.5f}".format(energy))

    # for istage in tqdm(range(nstage), desc="annealing progress"):
    for istage in range(nstage):
        naccept = 0
        for _ in range(niter):
            # permutation
            e1 = rs.randint(m)
            e2 = rs.randint(m)

            a, b = u[e1], v[e1]
            c, d = u[e2], v[e2]

            sb_prime = sb.copy()
            sb_prime[[a, b]] = sb_prime[[a, b]] - wts[e1] + wts[e2]
            sb_prime[[c, d]] = sb_prime[[c, d]] + wts[e1] - wts[e2]

            if energy_func is not None:
                energy_prime = energy_func(sb_prime, s)
            elif energy_type == "sse":
                energy_prime = np.sum((sb_prime - s) ** 2)
            elif energy_type == "max":
                energy_prime = np.max(np.abs(sb_prime - s))
            elif energy_type == "mae":
                energy_prime = np.mean(np.abs(sb_prime - s))
            elif energy_type == "mse":
                energy_prime = np.mean((sb_prime - s) ** 2)
            elif energy_type == "rmse":
                energy_prime = np.sqrt(np.mean((sb_prime - s) ** 2))
            else:
                msg = (
                    "energy_type must be one of 'sse', 'max', "
                    "'mae', 'mse', or 'rmse'. "
                    "Received: {}.".format(energy_type)
                )
                raise ValueError(msg)

            # permutation acceptance criterion
            if energy_prime < energy or rs.rand() < np.exp(
                -(energy_prime - energy) / temp
            ):
                sb = sb_prime.copy()
                wts[[e1, e2]] = wts[[e2, e1]]
                energy = energy_prime
                if energy < energymin:
                    energymin = energy
                    wtsmin = wts.copy()
                naccept = naccept + 1

        # temperature update
        temp = temp * frac
        if verbose:
            print(
                "\nstage {:d}, temp {:.5f}, best energy {:.5f}, "
                "frac of accepted moves {:.3f}".format(
                    istage, temp, energymin, naccept / niter
                )
            )

    B = np.zeros((n, n))
    B[(u, v)] = wtsmin
    B = B + B.T

    return B, energymin

def strength_preserving_rand_sa_dir(
    A,
    rewiring_iter=10,
    nstage=100,
    niter=10000,
    temp=1000,
    frac=0.5,
    full_rand=False,
    energy_type="sse",
    energy_func=None,
    connected=True,
    verbose=False,
    seed=None,
):
    """
    Strength-preserving network randomization using simulated annealing.

    Randomize a directed weighted network, while preserving
    the in- and out-degree and strength sequences using simulated annealing.

    Parameters
    ----------
    A : (N, N) array-like
        Directed weighted connectivity matrix
    rewiring_iter : int, optional
        Rewiring parameter. Default = 10.
        Each edge is rewired approximately rewiring_iter times.
    nstage : int, optional
        Number of annealing stages. Default = 100.
    niter : int, optional
        Number of iterations per stage. Default = 10000.
    temp : float, optional
        Initial temperature. Default = 1000.
    frac : float, optional
        Fractional decrease in temperature per stage. Default = 0.5.
    energy_type: str, optional
        Energy function to minimize. Can be either:
            'sse': Sum of squared errors between strength sequence vectors
                   of the original network and the randomized network
            'max': Maximum absolute error
            'mae': Mean absolute error
            'mse': Mean squared error
            'rmse': Root mean squared error
        Default = 'sse'.
    energy_func: callable, optional
        Callable with two positional arguments corresponding to
        two strength sequence numpy arrays that returns an energy value.
        Overwrites “energy_type”.
        See “energy_type” for specifying a predefined energy type instead.
    connected: bool, optional
        Whether to ensure connectedness of the randomized network.
        Default = True.
    verbose: bool, optional
        Whether to print status to screen at the end of every stage.
        Default = False.
    seed: float, optional
        Random seed. Default = None.

    full_rand: bool, optional
        Whether to perform a full randomization instead of Maslov&Sneppen rewiring.
        Only use when have very dense networks to save time!


    Returns
    -------
    B : (N, N) array-like
        Randomized connectivity matrix
    min_energy : float
        Minimum energy obtained by annealing

    Notes
    -----
    Uses Maslov & Sneppen rewiring model to produce a
    surrogate connectivity matrix, B, with the same
    size, density, and in- and out-degree sequences as A.
    The weights are then permuted to optimize the
    match between the strength sequences of A and B
    using simulated annealing.
    Both in- and out-strengths are preserved.

    This function is adapted from a function written in MATLAB
    by Richard Betzel.

    References
    ----------
    Misic, B. et al. (2015) Cooperative and Competitive Spreading Dynamics
    on the Human Connectome. Neuron.
    Rubinov, M. (2016) Constraints and spandrels of interareal connectomes.
    Nature Communications.
    Milisav, F. et al. (2024) A simulated annealing algorithm for
    randomizing weighted networks.
    """
    try:
        A = np.asarray(A)
    except TypeError as err:
        msg = "A must be array_like. Received: {}.".format(type(A))
        raise TypeError(msg) from err

    if frac > 1 or frac <= 0:
        msg = "frac must be between 0 and 1. " "Received: {}.".format(frac)
        raise ValueError(msg)

    rs = check_random_state(seed)

    n = A.shape[0]
    s_in = np.sum(A, axis=0)  # in-strengths of A
    s_out = np.sum(A, axis=1)  # out-strengths of A

    if full_rand:
        u, v = np.where(A > 0)
        w = A[(u, v)]
        np.random.shuffle(w)
        B = np.zeros_like(A)
        B[(u, v)] = w
    else:
        # Maslov & Sneppen rewiring
        if connected:
            B = bct.randmio_dir_connected(A, rewiring_iter, seed=seed)[0]
        else:
            B = bct.randmio_dir(A, rewiring_iter, seed=seed)[0]

    u, v = B.nonzero()  # nonzero indices of B
    wts = B[(u, v)]  # nonzero values of B
    m = len(wts)
    sb_in = np.sum(B, axis=0)  # in-strengths of B
    sb_out = np.sum(B, axis=1)  # out-strengths of B

    if energy_func is not None:
        energy = energy_func(s_in, sb_in) + energy_func(s_out, sb_out)
    elif energy_type == "sse":
        energy = np.sum((s_in - sb_in) ** 2) + np.sum((s_out - sb_out) ** 2)
    elif energy_type == "max":
        energy = np.max(np.abs(s_in - sb_in)) + np.max(np.abs(s_out - sb_out))
    elif energy_type == "mae":
        energy = np.mean(np.abs(s_in - sb_in)) + np.mean(np.abs(s_out - sb_out))
    elif energy_type == "mse":
        energy = np.mean((s_in - sb_in) ** 2) + np.mean((s_out - sb_out) ** 2)
    elif energy_type == "rmse":
        energy = np.sqrt(np.mean((s_in - sb_in) ** 2)) + np.sqrt(
            np.mean((s_out - sb_out) ** 2)
        )
    else:
        msg = (
            "energy_type must be one of 'sse', 'max', "
            "'mae', 'mse', or 'rmse'. Received: {}.".format(energy_type)
        )
        raise ValueError(msg)

    energymin = energy
    wtsmin = wts.copy()

    if verbose:
        print("\ninitial energy {:.5f}".format(energy))

    # for istage in tqdm(range(nstage), desc="annealing progress"):
    for istage in range(nstage):
        naccept = 0
        for _ in range(niter):
            # permutation
            e1 = rs.randint(m)
            e2 = rs.randint(m)

            a, b = u[e1], v[e1]
            c, d = u[e2], v[e2]

            sb_prime_in = sb_in.copy()
            sb_prime_out = sb_out.copy()
            sb_prime_in[b] = sb_prime_in[b] - wts[e1] + wts[e2]
            sb_prime_out[a] = sb_prime_out[a] - wts[e1] + wts[e2]
            sb_prime_in[d] = sb_prime_in[d] - wts[e2] + wts[e1]
            sb_prime_out[c] = sb_prime_out[c] - wts[e2] + wts[e1]

            if energy_func is not None:
                energy_prime = energy_func(sb_prime_in, s_in) + energy_func(
                    sb_prime_out, s_out
                )
            elif energy_type == "sse":
                energy_prime = np.sum((sb_prime_in - s_in) ** 2) + np.sum(
                    (sb_prime_out - s_out) ** 2
                )
            elif energy_type == "max":
                energy_prime = np.max(np.abs(sb_prime_in - s_in)) + np.max(
                    np.abs(sb_prime_out - s_out)
                )
            elif energy_type == "mae":
                energy_prime = np.mean(np.abs(sb_prime_in - s_in)) + np.mean(
                    np.abs(sb_prime_out - s_out)
                )
            elif energy_type == "mse":
                energy_prime = np.mean((sb_prime_in - s_in) ** 2) + np.mean(
                    (sb_prime_out - s_out) ** 2
                )
            elif energy_type == "rmse":
                energy_prime = np.sqrt(np.mean((sb_prime_in - s_in) ** 2)) + np.sqrt(
                    np.mean((sb_prime_out - s_out) ** 2)
                )
            else:
                msg = (
                    "energy_type must be one of 'sse', 'max', "
                    "'mae', 'mse', or 'rmse'. "
                    "Received: {}.".format(energy_type)
                )
                raise ValueError(msg)

            # permutation acceptance criterion
            if energy_prime < energy or rs.rand() < np.exp(
                -(energy_prime - energy) / temp
            ):
                sb_in = sb_prime_in.copy()
                sb_out = sb_prime_out.copy()
                wts[[e1, e2]] = wts[[e2, e1]]
                energy = energy_prime
                if energy < energymin:
                    energymin = energy
                    wtsmin = wts.copy()
                naccept = naccept + 1

        # temperature update
        temp = temp * frac
        if verbose:
            print(
                "\nstage {:d}, temp {:.5f}, best energy {:.5f}, "
                "frac of accepted moves {:.3f}".format(
                    istage, temp, energymin, naccept / niter
                )
            )

    B = np.zeros((n, n))
    B[(u, v)] = wtsmin

    return B, energymin

def test_sa_null(num_nodes=100, num_communities=4, plot=False,
                    null_kwargs={ 
                    'full_rand': True,
                    'rewiring_iter': 10,
                    'nstage': 20,
                    'niter': 5000,
                    'temp': 100,
                    'frac': 0.75,
                    'connected': False
                    }):
    
    def c_dense_network(num_nodes, num_communities=4):
        G = nx.DiGraph()
        community_labels = np.arange(num_nodes) % num_communities
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    if community_labels[i] == community_labels[j]:
                        weight = np.random.uniform(0.7, 1.0)
                    else:
                        weight = np.random.uniform(0.0, 0.3)
                    G.add_edge(i, j, weight=weight)
        return nx.to_numpy_array(G, weight='weight')
    
    G_mat = c_dense_network(num_nodes, num_communities)
    
    null, _ = strength_preserving_rand_sa_dir(G_mat, **null_kwargs)
    
    from scipy.stats import pearsonr
    corr_weights = pearsonr(G_mat.flatten(), null.flatten())[0]
    print(f"Weight Correlation (Original vs Null): {corr_weights:.4f}", flush=True)
    orig_strength = G_mat.sum(axis=0)
    null_strength = null.sum(axis=0)
    corr_strength = pearsonr(orig_strength, null_strength)[0]
    print(f"Strength Correlation (Original vs Null): {corr_strength:.4f}", flush=True)

    if plot:
        plt.figure(figsize=(4, 4))
        plt.plot(G_mat.sum(axis=0), null.sum(axis=0), 'o', c='grey', alpha=.9)
        plt.xlabel('Original Strength Sequence')
        plt.ylabel('Randomized Strength Sequence')

#########################################################
# MULTILAYER
#########################################################


def _rescaled_lap(W):
    L = laplacian(W)
    t = np.trace(L)
    return L / t if t > 0 else L

def _vn_entropy_rho(rho):
    evals = eigvalsh(rho)
    nz = evals[evals > 1e-12]
    if nz.size == 0:
        return 0.0
    return -float(np.sum(nz * np.log2(nz)))

def _js_distance(rho_a, rho_b, h_a, h_b):
    h_mix = _vn_entropy_rho(0.5 * (rho_a + rho_b))
    d_js = h_mix - 0.5 * (h_a + h_b)
    return float(np.sqrt(max(d_js, 0.0)))


def structural_reducibility(M, return_clusters=False):
    """
    Domenico et al. 2015 Nat Comms 
    Structural reducibility for a multilayer network.

    M : (N, N, L) ndarray. M[:, :, l] is the symmetric, non-negative adjacency
        matrix of layer l. Caller is responsible for any normalisation.

    Returns dict:
      D_JS    : (L, L) pairwise quantum Jensen-Shannon distance between layers
      Z       : scipy linkage matrix (Ward) on D_JS
      q_curve : (L,) values of q(C) at X = L, L-1, ..., 1 layers
      k_star  : optimal number of layers (1..L)
      chi     : reducibility (L - k_star) / (L - 1)
      h_A     : VN entropy of the aggregated single-layer graph
      hs      : (L,) per-layer entropies
      clusters_at_step : list of cluster-partitions per step (if return_clusters)
    """
    from scipy.spatial.distance import squareform
    from scipy.cluster.hierarchy import linkage

    M = np.asarray(M)
    if M.ndim != 3 or M.shape[0] != M.shape[1]:
        raise ValueError(f'Expected (N, N, L) array, got shape {M.shape}')
    L = M.shape[2]

    rhos = [_rescaled_lap(M[:, :, l]) for l in range(L)]
    hs = np.array([_vn_entropy_rho(r) for r in rhos])
    A = M.sum(axis=2)
    h_A = _vn_entropy_rho(_rescaled_lap(A))

    D = np.zeros((L, L))
    for i in range(L):
        for j in range(i + 1, L):
            D[i, j] = D[j, i] = _js_distance(rhos[i], rhos[j], hs[i], hs[j])

    Z = linkage(squareform(D, checks=False), method='ward')

    cluster_of_id = {l: [l] for l in range(L)}
    q_curve = np.zeros(L)
    clusters_at_step = []

    def _q_from_clusters(clusters):
        h_list = []
        for c in clusters:
            W_c = M[:, :, c].sum(axis=2)
            h_list.append(_vn_entropy_rho(_rescaled_lap(W_c)))
        H_bar = float(np.mean(h_list))
        return 1.0 - H_bar / h_A if h_A > 0 else 0.0

    clusters = list(cluster_of_id.values())
    q_curve[0] = _q_from_clusters(clusters)
    clusters_at_step.append([list(c) for c in clusters])

    for m in range(L - 1):
        a, b = int(Z[m, 0]), int(Z[m, 1])
        merged = cluster_of_id.pop(a) + cluster_of_id.pop(b)
        cluster_of_id[L + m] = merged
        clusters = list(cluster_of_id.values())
        q_curve[m + 1] = _q_from_clusters(clusters)
        clusters_at_step.append([list(c) for c in clusters])

    argmax_m = int(np.argmax(q_curve))
    k_star = L - argmax_m
    chi = (L - k_star) / (L - 1) if L > 1 else 0.0

    out = {
        'D_JS': D,
        'Z': Z,
        'q_curve': q_curve,
        'k_star': k_star,
        'chi': chi,
        'h_A': h_A,
        'hs': hs,
    }
    if return_clusters:
        out['clusters_at_step'] = clusters_at_step
    return out


########################################################
# PLOTTING
#########################################################

def plot_network(matrix, comms,
                      norm=True,
                      graph_type='undirected',
                      meta_scale=50,
                      scale=1,
                      community_layout='grid',
                      thresh=10,
                      max_edges=2000,
                      node_cmap='cubehelix',
                      edge_color='grey',
                      node_color_by='community',
                      nfactor=20,
                      node_size_range=(0.1, 175),
                      fig=None, ax=None,
                      layout='spring',
                      community_jitter=0.3,
                      connectionstyle="arc3,rad=0.5",
                      self_loops=False,
                      positions=None,
                      verbose=False):
    
    if norm == True:
        matrix = matrix.astype(float)
        min_val = np.nanmin(matrix)
        max_val = np.nanmax(matrix)
        if max_val > min_val:
            matrix = (matrix - min_val) / (max_val - min_val)

    if thresh is not None:
        flat_matrix = np.array(matrix).flatten()
        flat_matrix = flat_matrix[flat_matrix > 0]  # Only consider non-zero
        if len(flat_matrix) > 0:
            thresh_value = np.nanpercentile(flat_matrix, thresh)
            matrix[matrix < thresh_value] = 0
            min_val = matrix.min()
            max_val = matrix.max()
            if max_val > min_val:
                matrix = (matrix - min_val) / (max_val - min_val)

    if verbose:
        print(f'Preproc done', flush=True)

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    if graph_type == 'undirected':
        G = nx.from_numpy_array(matrix)
    else:
        G = nx.to_networkx_graph(matrix, create_using=nx.DiGraph)

    # Remove zero-weight edges
    zero_edges = [(u, v) for u, v in G.edges() if G[u][v].get('weight', 0) == 0]
    G.remove_edges_from(zero_edges)

    if self_loops == False:
        self_loop_edges = [(u, v) for u, v in G.edges() if u == v]
        G.remove_edges_from(self_loop_edges)

    if verbose:
        print(f'Graph has {G.number_of_edges()} edges after thresholding', flush=True)

    # limit total edges
    if max_edges is not None and G.number_of_edges() > max_edges:
        if verbose:
            print(f'Limiting to top {max_edges} edges...', flush=True)
        edge_list = [(u, v, G[u][v]['weight']) for u, v in G.edges()]
        edge_list.sort(key=lambda x: x[2], reverse=True)
        edges_to_keep = set([(u, v) for u, v, _ in edge_list[:max_edges]])
        edges_to_remove = [(u, v) for u, v in G.edges() if (u, v) not in edges_to_keep]
        G.remove_edges_from(edges_to_remove)
        if verbose:
            print(f'Reduced to {G.number_of_edges()} edges', flush=True)

    edges = list(G.edges())
    
    # If positions are provided, use them; otherwise compute new positions
    if positions is None:
        if verbose:
            print(f'Creating meta-graph for community layout', flush=True)

        num_comms = len(comms)
        meta_matrix = np.zeros((num_comms, num_comms))
        
        for i, comm_i in enumerate(comms):
            for j, comm_j in enumerate(comms):
                if i != j:
                    for node_i in comm_i:
                        for node_j in comm_j:
                            if G.has_edge(node_i, node_j):
                                meta_matrix[i, j] += G[node_i][node_j].get('weight', 1)
        
        meta_G = nx.from_numpy_array(meta_matrix)
        
        if community_layout == 'grid':
            grid_size = int(np.ceil(np.sqrt(num_comms)))
            spacing = 10
            centers = []
            for idx in range(num_comms):
                row = idx // grid_size
                col = idx % grid_size
                jitter_x = np.random.uniform(-spacing * community_jitter, spacing * community_jitter)
                jitter_y = np.random.uniform(-spacing * community_jitter, spacing * community_jitter)
                centers.append((col * spacing + jitter_x, -row * spacing + jitter_y))
        elif community_layout == 'spring':
            center_pos = nx.spring_layout(meta_G, scale=meta_scale, k=3, iterations=100, seed=None)
            centers = [center_pos[i] for i in range(num_comms)]
        elif community_layout == 'kamada_kawai':
            center_pos = nx.kamada_kawai_layout(meta_G, scale=meta_scale)
            centers = []
            for i in range(num_comms):
                x, y = center_pos[i]
                jitter_x = np.random.uniform(-5 * community_jitter, 5 * community_jitter)
                jitter_y = np.random.uniform(-5 * community_jitter, 5 * community_jitter)
                centers.append((x + jitter_x, y + jitter_y))
        elif community_layout == 'random':
            scale_factor = 30
            centers = [(np.random.uniform(-scale_factor, scale_factor),
                       np.random.uniform(-scale_factor, scale_factor))
                      for _ in range(num_comms)]
        elif community_layout == 'organic':
            center_pos = nx.spring_layout(meta_G, scale=meta_scale, k=2, iterations=100)
            centers = []
            for i in range(num_comms):
                x, y = center_pos[i]
                angle = np.arctan2(y, x)
                radius = np.sqrt(x**2 + y**2)
                radius *= np.random.uniform(0.5, 1.5)
                new_x = radius * np.cos(angle) + np.random.uniform(-3, 3)
                new_y = radius * np.sin(angle) + np.random.uniform(-3, 3)
                centers.append((new_x, new_y))
        else:
            raise ValueError("community_layout must be 'grid', 'spring', 'kamada_kawai', 'random', or 'organic'")
        
        pos = {}
        for center, comm in zip(centers, comms):
            subG = G.subgraph(comm)
            
            if layout == 'spring':
                subpos = nx.spring_layout(subG, scale=scale, k=.1, iterations=50)
            else:
                subpos = nx.kamada_kawai_layout(subG, scale=scale)
            
            for node, (x, y) in subpos.items():
                pos[node] = (x + center[0], y + center[1])
    else:
        pos = positions
        centers = None
    
    degree = dict(G.degree(weight='weight'))
    node_sizes_raw = [degree[node]*nfactor for node in G.nodes()]

    # Scale node sizes to specified range
    if node_size_range is not None and len(node_sizes_raw) > 0:
        min_range, max_range = node_size_range
        min_size = min(node_sizes_raw)
        max_size = max(node_sizes_raw)
        if max_size > min_size:
            node_sizes = [min_range + (s - min_size) / (max_size - min_size) * (max_range - min_range) 
                          for s in node_sizes_raw]
        else:
            node_sizes = [(min_range + max_range) / 4] * len(node_sizes_raw)  # midpoint if all equal
    else:
        if len(node_sizes_raw) > 0 and len(set(node_sizes_raw)) == 1:
            node_sizes = [node_sizes_raw[0] / 2] * len(node_sizes_raw)
        else:
            node_sizes = node_sizes_raw

    if node_color_by == 'degree':
        node_color = [degree[node] for node in G.nodes()]
    elif node_color_by == 'community':
        node_to_comm = {}
        for i, comm in enumerate(comms):
            for node in comm:
                node_to_comm[node] = i
        node_color = [node_to_comm[node] for node in G.nodes()]

    # Get edge weights
    if len(edges) > 0:
        weights = [G[u][v]['weight'] for u, v in edges]
        min_weight = min(weights)
        max_weight = max(weights)
        if max_weight > min_weight:
            alphas = [(w - min_weight) / (max_weight - min_weight) for w in weights]
        else:
            alphas = [0.5] * len(weights)
        # Ensure minimum visibility
        alphas = [max(0.3, a) for a in alphas]
    else:
        alphas = 0.5

    if verbose:
        print(f'Drawing {len(edges)} edges', flush=True)

    nx.draw_networkx_edges(G, pos, ax=ax, 
                           edgelist=edges,
                           edge_color=edge_color,
                           arrowsize=10, 
                           alpha=alphas,
                           node_size=node_sizes,
                           connectionstyle=connectionstyle,
                           width=1)
    
    if verbose:
        print(f'Drawing nodes', flush=True)

    nx.draw_networkx_nodes(G, pos, ax=ax,
                            node_size=node_sizes, 
                            node_color=node_color,
                            cmap=plt.get_cmap(node_cmap),
                            edgecolors='grey',
                            linewidths=.5,
                            alpha=1)
    
    ax.axis('off')
    return fig, ax, centers, pos




###########################################################
# COMMUNICATION MODELS
###########################################################

def compute_communication_models(s_conn, s_dist,
                                 norm=False,
                                 memory_SI=False,
                                 invert_SI=True,
                                 ensure_symm=True,
                                 models=['SP', 'NE', 'SI', 'CE', 'DE']):
    """
    Computes a spectrum of communication models
    
    from routing protocols, to navigation strategies, to broadcasting, to diffusion processes.
    
    Args:
        s_conn (np.ndarray): Structural connectivity matrix
            where s_conn[i,j] is the number of tracts between i and j.
        s_dist (np.ndarray): Distance matrix.
            where s_dist[i,j] is the Euclidean distance between i and j.
        norm (bool): If True, normalizes the matrix using given method.
            Default: False.
            'log_minmax': log-minmax normalization.
            'minmax': minmax normalization.
        memory_SI (bool): If True, random walker for SI remembers its last step
            Default: False.
        invert_SI (bool): If True, inverts SI to make commensurate with other models
            Default: False
        
    Returns:
        dict: Dictionary containing communication models.

            'SP': Shortest Path Efficiency.
            'NE': Navigation Efficiency.
            'SI': Search Information.
            'CE': Communicability.
            'DE': Diffusion Efficiency.
    """

    if norm == 'log_minmax':
        s_conn = log_minmax_normalisation(s_conn)
    elif norm == 'minmax':
        s_conn = minmax_normalisation(s_conn)

    models = {key: None for key in models}
    if 'SP' in models:
        models['SP'] = shortest_path_efficiency(s_conn)
    if 'NE' in models:
        models['NE'] = navigation_efficiency(s_conn, s_dist)
    if 'SI' in models:
        models['SI'] = search_information(s_conn, has_memory=memory_SI, invert=invert_SI)
    if 'CE' in models:
        models['CE'] = communicability(s_conn)
    if 'DE' in models:
        models['DE'] = diffusion_efficiency(s_conn)

    if ensure_symm:
        for key in models:
            models[key] = make_symmetric(models[key])
            np.fill_diagonal(models[key], 0.0)

    return models


def shortest_path_efficiency(s_conn):
    length = get_length(s_conn)
    SPL, _, _ = bct.distance_wei_floyd(length)
    SPE = 1. / SPL
    np.fill_diagonal(SPE, 0)
    return SPE
    

def navigation_wu(L, D, max_hops=None):
    """
    Navigation of connectivity length matrix L guided by nodal distance D

    Parameters:
    -----------
    L : numpy.ndarray
        Weighted/unweighted directed/undirected NxN SC matrix of connection lengths.
        L(i,j) is the strength-to-length remapping of the connection weight
        between i and j. L(i,j) = 0 denotes the lack of a connection between i and j.
    
    D : numpy.ndarray
        Symmetric NxN nodal distance matrix (e.g., Euclidean distance between node centroids)
    
    max_hops : int, optional
        Limits the maximum number of hops of navigation paths. Default is N (size of matrix).
    
    Returns:
    --------
    sr : float
        The success ratio (scalar) is the proportion of node pairs
        successfully reached by navigation.
    
    PL_bin : numpy.ndarray
        NxN matrix of binary navigation path length (i.e., number of hops in
        navigation paths). Infinite values indicate failed navigation paths.
    
    PL_wei : numpy.ndarray
        NxN matrix of weighted navigation path length (i.e., sum of connection 
        weights--as defined by L--along navigation path).
        Infinite values indicate failed navigation paths.
    
    PL_dis : numpy.ndarray
        NxN matrix of distance-based navigation path length (i.e., sum of connection 
        distances--as defined by D--along navigation path).
        Infinite values indicate failed navigation paths.
    
    paths : list of lists
        NxN list of nodes comprising navigation paths.
    
    Reference: Seguin et al. (2018) PNAS.
    """
    
    N = L.shape[0]
    
    if max_hops is None:
        max_hops = N
        
    paths = [[[] for _ in range(N)] for _ in range(N)]
    PL_bin = np.zeros((N, N))
    PL_wei = np.zeros((N, N))
    PL_dis = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            if i != j:
                curr_node = i
                last_node = curr_node
                target = j
                paths[i][j] = [curr_node]
                
                pl_bin = 0
                pl_wei = 0
                pl_dis = 0
                
                while curr_node != target:
                    neighbors = np.where(np.isfinite(L[curr_node, :]) & (L[curr_node, :] > 0))[0]
                    
                    if len(neighbors) == 0:
                        pl_bin = np.inf
                        pl_wei = np.inf
                        pl_dis = np.inf
                        break
                    
                    distances = D[target, neighbors]
                    min_index = np.argmin(distances)
                    next_node = neighbors[min_index]
                    
                    if next_node == last_node or pl_bin > max_hops:
                        pl_bin = np.inf
                        pl_wei = np.inf
                        pl_dis = np.inf
                        break
                    
                    paths[i][j].append(next_node)
                    pl_bin += 1
                    pl_wei += L[curr_node, next_node]
                    pl_dis += D[curr_node, next_node]
                    
                    last_node = curr_node
                    curr_node = next_node
                
                PL_bin[i, j] = pl_bin
                PL_wei[i, j] = pl_wei
                PL_dis[i, j] = pl_dis
    
    # Set diagonal elements to infinity
    np.fill_diagonal(PL_bin, np.inf)
    np.fill_diagonal(PL_wei, np.inf) 
    np.fill_diagonal(PL_dis, np.inf)
    
    # Calculate success ratio
    failed_paths = np.sum(np.isinf(PL_bin)) - N  # subtract diagonal
    total_paths = N * N - N  # total possible paths excluding self-connections
    sr = 1 - failed_paths / total_paths
    
    return sr, PL_bin, PL_wei, PL_dis, paths

def navigation_efficiency(s_conn, s_dist):
    length = get_length(s_conn)
    sr, PL_bin, PL_wei, PL_dis, paths = navigation_wu(length, s_dist)
    PL_eff = 1 / PL_dis
    return PL_eff

def search_information(s_conn, has_memory=False, invert=False):
    s_conn = np.array(s_conn, dtype=float)
    row_sums = s_conn.sum(axis=1)
    isolated = row_sums == 0
    if isolated.any():
        # give isolated nodes a tiny self-loop so diag(row_sums) is non-singular;
        # their paths to all other nodes will remain effectively inf
        s_conn = s_conn + np.diag(np.where(isolated, np.finfo(float).tiny, 0.0))
    SI = bct.search_information(s_conn, transform='inv', has_memory=has_memory)
    np.fill_diagonal(SI, 0.0)
    if invert:
        SI = 2**(-SI)
    return SI

def communicability(adjacency_matrix, alpha=1):
    """
    Computes the communicability of the network, with the option to be scaled by a decay rate factor 'alpha'.
    The alpha factor modulates the decay rate of walks, with smaller values leading
    to quicker subsidence. Alpha should be in the range (0, spectral radius of A).
    Works for binary, weighted, and directed graphs. See [1] for more details.

    Note from Gorka Zamora-lopez on directed graphs: For the adjacency matrix 'A', A_{ij} = 1 indicates j --> i,
    which is the opposite of the conventional graph theory notation. If your adjacency
    matrices follow the graph theory convention, ensure to transpose it first.

    Args:
        adjacency_matrix (np.ndarray): A square and nonnegative matrix representing network's structure.
        alpha (float, optional): The scaling factor. Defaults to 1, meaning that no scaling is applied.

    Returns:
        np.ndarray: The (scaled) communicability matrix. Shape: (N, N)

    References:
        [1] https://arxiv.org/abs/2307.02449

    # https://github.com/kuffmode/YANAT

    """
    from scipy.linalg import expm

    if adjacency_matrix.shape[1] != adjacency_matrix.shape[0]:
        raise ValueError("The adjacency matrix must be square. Like Spongebob's pants.")

    if np.any(adjacency_matrix < 0.0):
        raise ValueError(
            "The adjacency matrix contains negative values. Unless we figure out some stuff, only nonnegative matrices are allowed."
        )

    communicability_matrix: np.ndarray = expm(adjacency_matrix * alpha)

    np.fill_diagonal(communicability_matrix, 0.0)
    return communicability_matrix


def sar_model(adjacency_matrix, omega, normalize=False):
    """
    Computes the analytical covariance matrix for the spatial autoregressive (SAR) model.

    The SAR model considers each region's activity as a weighted sum of fluctuations from
     other regions, adjusted by a spatial influence factor 'alpha', plus a unit-variance Gaussian noise.
     The covariance matrix is analytically derived as the inverse of (I - alpha * A) times its transpose.
     See [1].

    Args:
        adjacency_matrix (np.ndarray): A square, nonnegative, and symmetric matrix representing network's structure.
        omega (float): The spatial influence factor, should be less than the spectral radius of 'adjacency_matrix'.
        The smaller the decay rate, the quicker it assumes the walks to be subsiding.
        normalize (bool, optional): If True, normalizes the matrix using strength normalization described by [2]. Defaults to False.

    Returns:
        np.ndarray: The covariance matrix of the SAR model. Shape: (N, N)

    References:
        [1] https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003530
        [2] https://royalsocietypublishing.org/doi/full/10.1098/rsif.2008.0484

    # https://github.com/kuffmode/YANAT
    """
    if normalize:
        row_sum = adjacency_matrix.sum(1)
        neg_sqrt = np.power(row_sum, -0.5)
        square_sqrt = np.diag(neg_sqrt)
        adjacency_matrix = square_sqrt @ adjacency_matrix @ square_sqrt

    N = adjacency_matrix.shape[0]
    sar = np.linalg.inv(np.eye(N) - omega * adjacency_matrix) @ np.linalg.inv(
        np.eye(N) - omega * adjacency_matrix.T
    )
    return sar

def lam(adjacency_matrix, alpha, normalize=False):
    """
    Computes the influence matrix for the Linear Attenuation Model (LAM), which underpins
    the dynamics of Katz centrality and is similar to communicability, but with a linear
    discount on longer walks rather than an exponential one. The discount factor 'alpha'
    should be less than the spectral radius of the adjacency matrix. See [1] for more details.

    Note from Gorka Zamora-lopez on directed graphs: For the adjacency matrix 'A', A_{ij} = 1 indicates j --> i,
    which is the opposite of the conventional graph theory notation. If your adjacency
    matrices follow the graph theory convention, ensure to transpose it first.

    Args:
        adjacency_matrix (np.ndarray): A square and nonnegative matrix representing network's structure.
        alpha (float): the decay rate. The smaller the decay rate, the quicker it assumes the walks to be subsiding.
        normalize (bool, optional): If True, applies strength normalization [2] to the matrix. Defaults to False.

    Returns:
        np.ndarray: The influence matrix for LAM. Shape: (N, N)

    References:
        [1] https://arxiv.org/abs/2307.02449
        [2] https://royalsocietypublishing.org/doi/full/10.1098/rsif.2008.0484
    """

    if normalize:
        adjacency_matrix: np.ndarray = strength_normalisation(adjacency_matrix)

    N: int = adjacency_matrix.shape[0]
    I: np.ndarray = np.eye(N)

    # Check if the user has given a shit about the documentation.
    if adjacency_matrix.shape[1] != N:
        raise ValueError("The adjacency matrix must be square. Like Spongebob's pants.")

    if np.any(adjacency_matrix < 0.0):
        raise ValueError(
            "The adjacency matrix contains negative values. Unless we figure out some stuff, only nonnegative matrices are allowed."
        )

    # Do some skibidi bapbap.
    influence_matrix = solve(I - alpha * adjacency_matrix, I)
    return influence_matrix


def _mfpt(adjacency):
    n = len(adjacency)
    mfpt = np.full((n, n), np.inf)
    np.fill_diagonal(mfpt, 0.0)

    # For symmetric inputs use weakly-connected components; for directed
    # inputs use strongly-connected components so each sub-chain is
    # irreducible and has a unique stationary distribution.
    sym = np.allclose(adjacency, adjacency.T)
    if sym:
        G = nx.from_numpy_array((adjacency + adjacency.T) > 0)
        comps = nx.connected_components(G)
    else:
        G = nx.from_numpy_array((adjacency > 0).astype(int), create_using=nx.DiGraph)
        comps = nx.strongly_connected_components(G)

    for comp in comps:
        comp = sorted(comp)
        if len(comp) <= 1:
            continue
        idx = np.array(comp)
        A_sub = adjacency[np.ix_(idx, idx)]
        row_sums = A_sub.sum(axis=1)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)  # sink nodes stay put
        P_sub = A_sub / row_sums[:, None]
        m = len(idx)
        D, V = np.linalg.eig(P_sub.T)
        aux = np.abs(D - 1)
        k = int(np.argmin(aux))
        w = np.real(V[:, k])
        # eigenvectors can be returned with arbitrary sign; flip so the
        # stationary distribution is non-negative before normalising.
        if w.sum() < 0:
            w = -w
        w = w / np.sum(w)
        W = np.repeat(np.atleast_2d(w), m, axis=0)
        Z = np.linalg.inv(np.eye(m) - P_sub + W)
        mfpt[np.ix_(idx, idx)] = np.real((np.repeat(np.atleast_2d(np.diag(Z)), m, axis=0) - Z) / W)

    return mfpt

def diffusion_efficiency(s_conn):
    mfpt = _mfpt(np.array(s_conn, dtype=float))
    DE = np.where(np.isinf(mfpt), 0.0, 1.0 / np.where(mfpt == 0, np.inf, mfpt))
    np.fill_diagonal(DE, 0)
    return DE