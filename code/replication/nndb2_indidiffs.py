import os
import sys
import gc
import itertools
import functools
print = functools.partial(print, flush=True)

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

import numpy as np
import pandas as pd
from scipy.io import loadmat
from joblib import Parallel, delayed

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
sys.path.insert(0, root) # indidiff_utils and the IG_phiid package live at the repo root
sys.path.insert(0, here) # replication misc.py must shadow the one at the root
sys.path.append(os.path.join(root, 'IG_phiid')) # gaussian_phiid bare-imports its siblings

import indidiff_utils as idu
import misc
from IG_phiid.gaussian_phiid import ig_phiid_gaussian
from IG_phiid.gaussian_utils import get_cov

parcellation = misc.parcellation
n_regions = misc.n_regions
n_atoms = 16
base_dir = os.path.join(here, 'nndb2_results') # anchored to replication/, not the launch dir
main_dir = f'{base_dir}/main'
out_dir = f'{base_dir}/indidiffs/{parcellation}'
data_dir = f'/data/nndb2/parcellated/{parcellation}'
subjects = misc.subjects
n_cores = 32
tau = misc.tau
sfreq = misc.sfreq
chunk_size = 500

atoms = misc.atoms

triu = np.triu_indices(n_regions, k=1)

regions = misc.regions
pairs = [(min(i, j), max(i, j)) for i, j in itertools.combinations(regions, 2)]


def _compute_phiid_pair(data_ij, t):
    """Time-averaged phiid for one pair. Returns (n_atoms,) or NaN array."""
    try:
        cov, vecs = get_cov(data_ij, t=t, get_vecs=True)
        phiid = ig_phiid_gaussian(cov, pointwise=False, as_dict=False, data=vecs)
        return np.asarray(phiid, dtype=np.float32)
    except Exception:
        return np.full(n_atoms, np.nan, dtype=np.float32)


def _build_phiid_matrix(data):
    """Compute time-averaged phiid matrix from (n_regions, n_time) data."""
    n_pairs = len(pairs)
    mat = np.full((n_regions, n_regions, n_atoms), np.nan, dtype=np.float32)

    for chunk_start in range(0, n_pairs, chunk_size):
        chunk = pairs[chunk_start: chunk_start + chunk_size]
        results = Parallel(n_jobs=n_cores)(
            delayed(_compute_phiid_pair)(data[list(p), :], tau)
            for p in chunk
        )
        for (i, j), res in zip(chunk, results):
            mat[i, j, :] = res
            mat[j, i, :] = res

        done = chunk_start + len(chunk)
        if done % 50000 == 0 or done == n_pairs:
            print(f'    {done}/{n_pairs} pairs done')

    return mat


def compute_and_save_split_half(perp):
    """
    Load full timeseries, bandpass, split at midpoint, compute phiid for each
    half (time-averaged), and save as phiid_matrix_half1.npy / half2.npy.
    Skips if both files already exist.
    """
    sub_dir = f'{main_dir}/{parcellation}/sub-{perp}'
    os.makedirs(sub_dir, exist_ok=True)
    path1 = f'{sub_dir}/phiid_matrix_half1.npy'
    path2 = f'{sub_dir}/phiid_matrix_half2.npy'

    if os.path.exists(path1) and os.path.exists(path2):
        print(f'sub-{perp}: split-half already exists, skipping')
        return

    print(f'sub-{perp}: computing split-half phiid...')
    data = loadmat(f'{data_dir}/sub-{perp}/full_ts.mat')['data']
    if data.shape[0] != n_regions:
        raise ValueError(f'sub-{perp}: got {data.shape[0]} regions, expected {n_regions}')
    data = misc.bandpass_filter(data, sfreq)

    mid = data.shape[1] // 2
    h1 = data[:, :mid]
    h2 = data[:, mid:]

    print(f'  half 1 ({h1.shape[1]} frames)...')
    mat1 = _build_phiid_matrix(h1)
    np.save(path1, mat1)
    del mat1; gc.collect()

    print(f'  half 2 ({h2.shape[1]} frames)...')
    mat2 = _build_phiid_matrix(h2)
    np.save(path2, mat2)
    del mat2; gc.collect()

    print(f'sub-{perp}: saved')


def load_half(perp, atom_indices, half):
    mat = np.load(f'{main_dir}/{parcellation}/sub-{perp}/phiid_matrix_half{half}.npy')
    out = mat[:, :, atom_indices].sum(axis=2).astype(np.float64)
    return out


def load_all_subjects(atom_indices):
    h1_list, h2_list, valid = [], [], []
    for perp in subjects:
        try:
            h1_list.append(load_half(perp, atom_indices, 1))
            h2_list.append(load_half(perp, atom_indices, 2))
            valid.append(perp)
        except Exception as e:
            print(f'  sub-{perp} skipped: {e}')
    return np.stack(h1_list), np.stack(h2_list), np.array(valid)


def get_valid_subjects():
    """Subjects with both split-half phiid matrices on disk."""
    valid = []
    for perp in subjects:
        p1 = f'{main_dir}/{parcellation}/sub-{perp}/phiid_matrix_half1.npy'
        p2 = f'{main_dir}/{parcellation}/sub-{perp}/phiid_matrix_half2.npy'
        if os.path.exists(p1) and os.path.exists(p2):
            valid.append(perp)
    return np.array(valid)


networks_dir = f'{base_dir}/networks/{parcellation}'


def compute_community_prob_matrix(atom_name):
    """
    For each edge (i, j), fraction of subjects where nodes i and j
    share the same community assignment (from networks.py output).
    Returns (n_regions, n_regions) float array, or None if no files found.
    """
    ci_dir = f'{networks_dir}/community_assignments'
    all_ci = []
    for perp in subjects:
        path = f'{ci_dir}/ci_{atom_name}_sub-{perp}.npy'
        if os.path.exists(path):
            all_ci.append(np.load(path))
    if not all_ci:
        print(f'  no community files found for {atom_name}, skipping')
        return None
    all_ci = np.stack(all_ci)
    same = (all_ci[:, :, None] == all_ci[:, None, :])
    return same.mean(axis=0)


if __name__ == '__main__':
    os.makedirs(out_dir, exist_ok=True)

    print('=== split-half phiid ===')
    for perp in subjects:
        try:
            compute_and_save_split_half(perp)
        except Exception as e:
            print(f'sub-{perp}: FAILED - {e}')

    print('\n=== identifiability & ICC ===')
    icc_records = []
    identi_records = []

    for atom_name, atom_indices in atoms.items():
        print(f'\n--- {atom_name} ---')

        icc_path = f'{out_dir}/icc_{atom_name}.npy'
        identi_path = f'{out_dir}/identi_{atom_name}.npy'
        iself_null_path = f'{out_dir}/iself_null_{atom_name}.npy'
        iother_null_path = f'{out_dir}/iother_null_{atom_name}.npy'
        idiff_null_path = f'{out_dir}/idiff_null_{atom_name}.npy'

        icc_cached = os.path.exists(icc_path)
        identi_cached = os.path.exists(identi_path)
        null_cached = (os.path.exists(iself_null_path)
                       and os.path.exists(iother_null_path)
                       and os.path.exists(idiff_null_path))

        if icc_cached and identi_cached:
            valid_subjects = get_valid_subjects()
            half1 = half2 = None
            print(f'  {len(valid_subjects)} subjects (data not loaded, using cached outputs)')
        else:
            half1, half2, valid_subjects = load_all_subjects(atom_indices)
            print(f'  {len(valid_subjects)} subjects loaded')

        if icc_cached:
            print('  ICC cached, loading')
            icc = np.load(icc_path)
        else:
            print('  computing ICC...')
            icc = idu.compute_icc_matrix(half1, half2)
            np.save(icc_path, icc.astype(np.float32))

        nodal_icc = np.nanmean(icc, axis=1)
        for reg, val in enumerate(nodal_icc):
            icc_records.append({'info': atom_name, 'region': reg, 'icc': float(val)})

        if identi_cached:
            print('  identifiability cached, loading')
            identi = np.load(identi_path).astype(np.float64)
        else:
            print('  computing identifiability...')
            half1_flat = half1[:, triu[0], triu[1]]
            half2_flat = half2[:, triu[0], triu[1]]
            identi = idu.compute_identifiability_matrix(half1_flat, half2_flat)
            np.save(identi_path, identi.astype(np.float32))
            del half1_flat, half2_flat

        iself, iother, idiff = idu.identifiability_stats(identi)
        for i, perp in enumerate(valid_subjects):
            identi_records.append({
                'subject': perp,
                'info': atom_name,
                'iself': float(iself[i]),
                'iother': float(iother[i]),
                'idiff': float(idiff[i]),
            })

        if null_cached:
            print('  permutation null cached, skipping')
        else:
            print('  computing identifiability permutation null...')
            iself_null, iother_null, idiff_null = idu.permutation_null_idiff(
                identi, n_perm=1000, seed=42)
            np.save(iself_null_path, iself_null.astype(np.float32))
            np.save(iother_null_path, iother_null.astype(np.float32))
            np.save(idiff_null_path, idiff_null.astype(np.float32))
            del iself_null, iother_null, idiff_null

        del half1, half2, identi
        gc.collect()

    print('\n=== community co-assignment probability ===')
    for atom_name in atoms:
        print(f'\n--- {atom_name} ---')
        prob_path = f'{out_dir}/community_prob_{atom_name}.npy'
        if os.path.exists(prob_path):
            print(f'  {atom_name}: already exists, skipping')
            continue
        prob = compute_community_prob_matrix(atom_name)
        if prob is not None:
            np.save(prob_path, prob.astype(np.float32))
            print(f'  saved community_prob_{atom_name}.npy')

    print('\nsaving CSVs...')
    pd.DataFrame(icc_records).to_csv(f'{out_dir}/nodal_icc.csv', index=False)
    pd.DataFrame(identi_records).to_csv(f'{out_dir}/identifiability.csv', index=False)
    print(f'  {out_dir}/nodal_icc.csv')
    print(f'  {out_dir}/identifiability.csv')
    print('\ndone.')
