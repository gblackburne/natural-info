import os
import sys
import gc
import numpy as np
import itertools
from scipy.io import loadmat, savemat
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import functools
print = functools.partial(print, flush=True)

from joblib import Parallel, delayed
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
n_cores = 40

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import misc

IG_path = os.path.join(os.getcwd(), "IG_phiid")
sys.path.append(IG_path)
from IG_phiid.gaussian_phiid import ig_phiid_gaussian
from IG_phiid.gaussian_utils import get_cov

atom_names = misc.phiid_atoms
n_atoms = len(atom_names)
atoms = misc.atoms
group_names = list(atoms.keys())
group_indices = [atoms[g] for g in group_names]
n_groups = len(group_names)

parcellation = misc.parcellation
n_regions = misc.n_regions
subjects = misc.subjects
regions = misc.regions
pairs = list(itertools.combinations(regions, 2))
pairs = [(min(i, j), max(i, j)) for i, j in pairs]
sfreq = misc.sfreq
tau = misc.tau

chunk_size = 500

def compute_phiid(data_ij, t, pointwise=True):
    """
    Compute PhiID for a given pair of regions.

    Parameters
    ----------
    data_ij : np.ndarray, shape (2, n_time)
    t : int  lag (tau)
    pointwise : bool
        True  -> return (16, n_time-t)
        False -> return (16,)
    """
    n_r, n_t = data_ij.shape
    try:
        cov, vecs = get_cov(data_ij, t=t, get_vecs=True)
        phiid = ig_phiid_gaussian(cov, pointwise=pointwise, as_dict=False, data=vecs)
        return phiid
    except Exception:
        print("PhiID computation failed, returning NaNs")
        if pointwise:
            return np.full((n_atoms, n_t - t), np.nan)
        else:
            return np.full(n_atoms, np.nan)


def process_subject(perp):
    print(f'Processing subject {perp}')

    data_path = f'/data/gblackburne/nndb/data/parcellated/{parcellation}/sub-{perp}/full_ts.mat'
    data = loadmat(data_path)['data'] # (n_regions, n_samples)
    data = misc.bandpass_filter(data, sfreq)

    n_samples = data.shape[1]
    n_time = n_samples - tau # length of pointwise time axis

    pearson_matrix = np.corrcoef(data)

    # Accumulation arrays
    phiid_matrix = np.full((n_regions, n_regions, n_atoms), np.nan, dtype=np.float32)
    region_sum   = np.zeros((n_regions, n_time, n_groups), dtype=np.float64)
    region_count = np.zeros((n_regions, n_time, n_groups), dtype=np.int32)

    n_pairs = len(pairs)
    for chunk_start in range(0, n_pairs, chunk_size):
        chunk = pairs[chunk_start: chunk_start + chunk_size]

        results = Parallel(n_jobs=n_cores)(
            delayed(compute_phiid)(data[list(pair), :], tau, pointwise=True)
            for pair in chunk
        )

        for (i, j), res in zip(chunk, results):
            time_avg = np.nanmean(res, axis=1) # (n_atoms,)
            phiid_matrix[i, j, :] = time_avg.astype(np.float32)
            phiid_matrix[j, i, :] = time_avg.astype(np.float32)

            res_T = res.T  # (n_atoms, n_time) -> (n_time, n_atoms)
            for g, idx in enumerate(group_indices):
                grp_val = res_T[:, idx].sum(axis=1)
                valid = ~np.isnan(grp_val)
                region_sum[i, :, g] += np.where(valid, grp_val, 0.0)
                region_sum[j, :, g] += np.where(valid, grp_val, 0.0)
                region_count[i, :, g] += valid
                region_count[j, :, g] += valid

        del results
        gc.collect()

        pairs_done = chunk_start + len(chunk)
        if pairs_done % 5000 == 0 or pairs_done == n_pairs:
            print(f'  sub-{perp}: {pairs_done}/{n_pairs} pairs done')

    # Regional nanmean: (n_regions, n_time, n_groups)
    with np.errstate(invalid='ignore'):
        phiid_regional = np.where(region_count > 0,
                                  region_sum / np.maximum(region_count, 1),
                                  np.nan)

    out_dir = f'results/main/{parcellation}/sub-{perp}'
    out_dir_dynamic = f'results/main_dynamic/{parcellation}/sub-{perp}'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_dynamic, exist_ok=True)

    np.save(f'{out_dir}/pearson_matrix.npy', pearson_matrix)
    np.save(f'{out_dir}/phiid_matrix.npy', phiid_matrix)
    np.save(f'{out_dir}/atom_names.npy', np.array(atom_names))
    savemat(f'{out_dir_dynamic}/phiid_regional.mat',
            {g: phiid_regional[:, :, i].astype(np.float32)
             for i, g in enumerate(group_names)},
            do_compression=True)

    print(f'Completed subject {perp}')


for perp in subjects:
    process_subject(perp)

def compute_group_phiid_matrix():
    out_path = f'results/main/{parcellation}/group_phiid_matrix.npy'
    if os.path.exists(out_path):
        print('Group phiid matrix already exists, skipping.')
        return
    print('Computing group-average phiid matrices...')
    group_phiid_matrix = {}
    for atom_name, atom_indices in atoms.items():
        matrices = [np.load(f'results/main/{parcellation}/sub-{perp}/phiid_matrix.npy')[:, :, atom_indices].sum(axis=2).astype(np.float64)
                    for perp in subjects
                    if os.path.exists(f'results/main/{parcellation}/sub-{perp}/phiid_matrix.npy')]
        if matrices:
            group_phiid_matrix[atom_name] = np.nanmean(np.stack(matrices), axis=0)
    np.save(out_path, group_phiid_matrix)
    print(f'Saved to {out_path}')

compute_group_phiid_matrix()


def compute_group_phiid_std():
    out_path = f'results/main/{parcellation}/group_phiid_std.npy'
    if os.path.exists(out_path):
        print('Group phiid std already exists, skipping.')
        return
    print('Computing group-std phiid matrices...')
    group_phiid_std = {}
    for atom_name, atom_indices in atoms.items():
        matrices = [
            np.load(f'results/main/{parcellation}/sub-{perp}/phiid_matrix.npy')
              [:, :, atom_indices].sum(axis=2).astype(np.float64)
            for perp in subjects
            if os.path.exists(f'results/main/{parcellation}/sub-{perp}/phiid_matrix.npy')
        ]
        if matrices:
            group_phiid_std[atom_name] = np.nanstd(np.stack(matrices), axis=0, ddof=1)
    np.save(out_path, group_phiid_std)
    print(f'Saved to {out_path}')

compute_group_phiid_std()


def compute_group_phiid_regions():
    out_path = f'results/main/{parcellation}/group_phiid_regions.npy'
    if os.path.exists(out_path):
        print('Group phiid regions already exists, skipping.')
        return
    print('Computing group-average phiid regional means...')
    group_phiid_region = {}
    for atom_name, atom_indices in atoms.items():
        matrices = []
        for perp in subjects:
            path = f'results/main/{parcellation}/sub-{perp}/phiid_matrix.npy'
            if os.path.exists(path):
                mat = np.load(path)[:, :, atom_indices].sum(axis=2).astype(np.float64)
                matrices.append(np.nanmean(mat, axis=0))
        if matrices:
            group_phiid_region[atom_name] = np.nanmean(np.stack(matrices), axis=0)
    np.save(out_path, group_phiid_region)
    print(f'Saved to {out_path}')

compute_group_phiid_regions()

print('Your time has come, run to the hills')
print('Live your life as though every act were to become a universal law')

