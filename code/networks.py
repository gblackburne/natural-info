import os
import sys
import gc
import warnings
import functools
print = functools.partial(print, flush=True)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['TQDM_DISABLE'] = '1'

import tqdm as _tqdm_module
_real_tqdm_init = _tqdm_module.tqdm.__init__
def _silent_tqdm_init(self, *args, **kwargs):
    kwargs['disable'] = True
    _real_tqdm_init(self, *args, **kwargs)
_tqdm_module.tqdm.__init__ = _silent_tqdm_init

import traceback
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_mutual_info_score

warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import network_utils as nu
import misc


parcellation = misc.parcellation
n_regions = misc.n_regions
base_dir = 'results'
main_dir = f'{base_dir}/main'
out_dir = f'{base_dir}/networks/{parcellation}'
atlas_csv = f'/data/gblackburne/fMRI_general/atlases/schaefer_extend_atlases/{parcellation}.csv'
subjects = misc.subjects
n_cores = 30
atoms = misc.atoms
sc_path = f'{out_dir}/SC.npy'
checkpoint_dir = f'{out_dir}/checkpoints'

null_kwargs = {
    'full_rand': True,
    'rewiring_iter': 10, 'nstage': 20, 'niter': 5000,
    'temp': 100, 'frac': 0.75, 'connected': False,
}

# consensus community detection parameters (group and individual)
consensus_kwargs = dict(
    rmin=0.75, space='linear',
    consensus_method='cpm', resolution_parameter=0.1,
    rnum_1=20, rnum_2=30,
    min_community_size=2,
)


scalar_measures = {
    'global_efficiency': nu.comp_efficiency,
    'von_neumann_entropy': nu.compute_von_ent,
    's_metric': nu.comp_s_metric,
}


def load_connectome(perp, atom_indices):
    mat = np.load(f'{main_dir}/{parcellation}/sub-{perp}/phiid_matrix.npy')
    out = mat[:, :, atom_indices].sum(axis=2).astype(np.float64)
    if np.isnan(out).any():
        min_finite = np.nanmin(out)
        out = np.where(np.isnan(out), min_finite, out)
    return out

def load_yeo_labels():
    df = pd.read_csv(atlas_csv)
    _, encoded = np.unique(df['network'].values, return_inverse=True)
    return encoded

def _scalar_df(perp, atom_name, measure, value):
    return pd.DataFrame([{
        'subject': perp, 'info': atom_name,
        'measure': measure, 'region': 'global', 'value': value,
    }])

def _nodal_df(perp, atom_name, measure, values):
    return pd.DataFrame({
        'subject': perp,
        'info': atom_name,
        'measure': measure,
        'region': np.arange(len(values)),
        'value': np.asarray(values, dtype=np.float32),
    })


def run_scalar_topology(perp, atom_name, W):
    results = {}
    for name, func in scalar_measures.items():
        try:
            val = float(func(W))
        except Exception as e:
            print(f'  ERROR sub-{perp} | {atom_name} | scalar {name}: {e}')
            val = np.nan
        results[f'scalar_{name}'] = _scalar_df(perp, atom_name, name, val)
    return results


def run_community_measures(perp, atom_name, W, group_ci, yeo_int, n_jobs=1):
    results = {}

    try:
        ci = nu.multires_consensus_comdet(
            W, n_jobs=n_jobs, **consensus_kwargs
        ).astype(int)
    except Exception as e:
        print(f'  ERROR sub-{perp} | {atom_name} | consensus comdet: {e}')
        ci = np.zeros(n_regions, dtype=int)

    ci_dir = f'{out_dir}/community_assignments'
    os.makedirs(ci_dir, exist_ok=True)
    np.save(f'{ci_dir}/ci_{atom_name}_sub-{perp}.npy', ci)

    for partition_name, partition in [('individual', ci), ('group', group_ci)]:
        try:
            q = float(nu.comp_mod(W, partition))
        except Exception as e:
            print(f'  ERROR sub-{perp} | {atom_name} | modularity_{partition_name}: {e}')
            q = np.nan
        results[f'community_modularity_{partition_name}'] = _scalar_df(
            perp, atom_name, f'modularity_{partition_name}', q)

        try:
            pc = nu.comp_partC(W, partition)
        except Exception as e:
            print(f'  ERROR sub-{perp} | {atom_name} | participation_coeff_{partition_name}: {e}')
            pc = np.full(n_regions, np.nan)
        results[f'community_participation_coeff_{partition_name}'] = _nodal_df(
            perp, atom_name, f'participation_coeff_{partition_name}', pc)

    try:
        ami_g = float(adjusted_mutual_info_score(ci, group_ci.astype(int)))
    except Exception as e:
        print(f'  ERROR sub-{perp} | {atom_name} | ami_vs_group: {e}')
        ami_g = np.nan
    results['community_ami_vs_group'] = _scalar_df(perp, atom_name, 'ami_vs_group', ami_g)

    try:
        ami_y = float(adjusted_mutual_info_score(ci, yeo_int.astype(int)))
    except Exception as e:
        print(f'  ERROR sub-{perp} | {atom_name} | ami_vs_yeo: {e}')
        ami_y = np.nan
    results['community_ami_vs_yeo'] = _scalar_df(perp, atom_name, 'ami_vs_yeo', ami_y)

    return results


def _group_atom_pipeline(atom_name, atom_indices):
    """Load precomputed group average and run community detection."""
    group_ci_path = f'{out_dir}/group_ci_{atom_name}.npy'
    if os.path.exists(group_ci_path):
        try:
            group_ci = np.load(group_ci_path).astype(int)
            print(f'[group] {atom_name}: loaded from cache ({len(np.unique(group_ci))} communities)')
            return atom_name, atom_indices, group_ci
        except Exception as e:
            print(f'[group] {atom_name}: corrupt cache ({e}), recomputing')
            try:
                os.remove(group_ci_path)
            except OSError:
                pass

    group_phiid_matrix = np.load(
        f'{main_dir}/{parcellation}/group_phiid_matrix.npy', allow_pickle=True
    ).item()
    print(f'[group] loaded precomputed group average for {atom_name}')
    raw = group_phiid_matrix[atom_name]
    if np.isnan(raw).any():
        raw = np.where(np.isnan(raw), np.nanmin(raw), raw)
    W = nu.rank_normalise(raw)
    np.fill_diagonal(W, 0)

    print(f'[group] clustering {atom_name}...')
    group_ci = nu.multires_consensus_comdet(
        W, n_jobs=n_cores, **consensus_kwargs
    ).astype(int)
    np.save(group_ci_path, group_ci)
    print(f'[group] {atom_name}: {len(np.unique(group_ci))} communities')

    return atom_name, atom_indices, group_ci


def _load_subject(perp, atom_indices, atom_name):
    try:
        W_raw = load_connectome(perp, atom_indices)
    except Exception as e:
        print(f'  sub-{perp} {atom_name}: load failed - {e}')
        return None
    W = nu.rank_normalise(W_raw)
    np.fill_diagonal(W, 0)
    return W


def _ckpt_path(phase, atom_name, perp):
    return f'{checkpoint_dir}/{phase}_{atom_name}_sub-{perp}.pkl'

def _try_load_checkpoint(path):
    if not os.path.exists(path):
        return None, False
    try:
        with open(path, 'rb') as f:
            return pickle.load(f), True
    except Exception as e:
        print(f'  corrupt checkpoint {path}: {e} -- recomputing')
        try:
            os.remove(path)
        except OSError:
            pass
        return None, False

def _save_checkpoint(path, results):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(results, f)
    os.replace(tmp, path)


def process_subject_first(perp, atom_indices, atom_name):
    ckpt = _ckpt_path('first', atom_name, perp)
    data, ok = _try_load_checkpoint(ckpt)
    if ok:
        print(f'sub-{perp} | {atom_name} (first) - cached')
        return data
    print(f'sub-{perp} | {atom_name} (first)')
    W = _load_subject(perp, atom_indices, atom_name)
    if W is None:
        return {}
    results = run_scalar_topology(perp, atom_name, W)
    del W
    gc.collect()
    _save_checkpoint(ckpt, results)
    return results


def process_subject_second(perp, atom_indices, atom_name, group_ci, yeo_int, n_jobs):
    ckpt = _ckpt_path('second', atom_name, perp)
    data, ok = _try_load_checkpoint(ckpt)
    if ok:
        print(f'sub-{perp} | {atom_name} (second) - cached')
        return data
    print(f'sub-{perp} | {atom_name} (second)')
    W = _load_subject(perp, atom_indices, atom_name)
    if W is None:
        return {}
    results = run_community_measures(perp, atom_name, W, group_ci, yeo_int, n_jobs=n_jobs)
    del W
    gc.collect()
    _save_checkpoint(ckpt, results)
    return results


def _r2_ols(y, x):
    """OLS R2 for y ~ 1 + x (both pre-masked 1D arrays)."""
    X = np.column_stack([np.ones(len(x)), x])
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

def _r2_for_pred(phiid_vecs, pred_vec):
    r2s = []
    for pv in phiid_vecs:
        mask = np.isfinite(pv) & np.isfinite(pred_vec)
        r2s.append(_r2_ols(pv[mask], pred_vec[mask]) if mask.sum() >= 100 else np.nan)
    return r2s

def _gen_null_cm(cm_mat, seed, **null_kwargs):
    null, _ = nu.strength_preserving_rand_sa_undir(cm_mat, seed=seed, **null_kwargs)
    return null


def _direct_indirect_subject(perp, atom_indices, atom_name, direct_mask, indirect_mask):
    try:
        W = load_connectome(perp, atom_indices)
    except Exception as e:
        print(f'  sub-{perp} {atom_name}: load failed - {e}')
        return []
    np.fill_diagonal(W, 0)
    W = nu.rank_normalise(W)
    triu = np.triu_indices(n_regions, k=1)
    w_utri = W[triu]
    rows = []
    for conn_type, mask in [('direct', direct_mask), ('indirect', indirect_mask)]:
        vals = w_utri[mask]
        rows.append({
            'subject': perp, 'atom': atom_name,
            'connection_type': conn_type,
            'value': float(np.nanmean(vals)) if mask.any() else np.nan,
        })
    return rows


def compute_direct_indirect_analysis():
    print('\n--- direct vs indirect connection analysis ---')
    if not os.path.exists(sc_path):
        print(f'  SC matrix not found at {sc_path} - skipping')
        return
    s_conn = np.load(sc_path)
    np.fill_diagonal(s_conn, 0)
    triu = np.triu_indices(n_regions, k=1)
    sc_utri = s_conn[triu]
    direct_mask = sc_utri > 0
    indirect_mask = sc_utri == 0
    print(f'  {direct_mask.sum():,} direct edges, {indirect_mask.sum():,} indirect edges')

    records = []
    for atom_name, atom_indices in atoms.items():
        print(f'  {atom_name}...')
        results = Parallel(n_jobs=n_cores)(
            delayed(_direct_indirect_subject)(perp, atom_indices, atom_name, direct_mask, indirect_mask)
            for perp in subjects
        )
        for subj_rows in results:
            records.extend(subj_rows)

    df = pd.DataFrame(records)
    out_path = f'{out_dir}/direct_indirect.csv'
    df.to_csv(out_path, index=False)
    print(f'saved {out_path} ({len(df):,} rows)')


def compute_matrix_r2_tests(n_nulls=1000):
    print('\n--- matrix R2 tests ---')
    cm_path = f'{out_dir}/CM.npy'
    if not os.path.exists(cm_path):
        print(f'  CM.npy not found at {cm_path} - skipping')
        return

    triu = np.triu_indices(n_regions, k=1)

    def utri(M):
        return M[triu]

    print('loading communication models...')
    cms = np.load(f'{out_dir}/CM.npy', allow_pickle=True).item()

    # raw phiids and raw cms
    cm_null_vecs = {}
    for cm_name, cm_mat in cms.items():
        null_path = f'{out_dir}/CM_nulls_{cm_name}.npy'
        if os.path.exists(null_path):
            try:
                print(f'  loading nulls for {cm_name}...')
                cm_null_vecs[cm_name] = list(np.load(null_path))
            except Exception as e:
                print(f'  corrupt null file {null_path}: {e} -- regenerating')
                os.remove(null_path)
        else:
            print(f'  generating {n_nulls} nulls for {cm_name}...')
            null_mats = Parallel(n_jobs=n_cores)(
                delayed(_gen_null_cm)(cm_mat, k, **null_kwargs) for k in range(n_nulls)
            )
            null_arr = np.array([utri(m) for m in null_mats], dtype=np.float32)
            np.save(null_path, null_arr)
            cm_null_vecs[cm_name] = list(null_arr)
            print(f'  {cm_name}: {n_nulls} nulls saved to {null_path}')

    cm_obs_vecs = {k: utri(v) for k, v in cms.items()}

    records = []

    for atom_name, atom_indices in atoms.items():
        print(f'\n  {atom_name}...')

        phiid_vecs = {}
        for perp in subjects:
            try:
                phiid = load_connectome(perp, atom_indices)
                np.fill_diagonal(phiid, 0)
                phiid_vecs[perp] = utri(phiid)
            except Exception as e:
                print(f'    sub-{perp} load failed: {e}')

        valid_perps = sorted(phiid_vecs)
        valid_vecs = [phiid_vecs[p] for p in valid_perps]

        for cm_name, obs_vec in cm_obs_vecs.items():
            obs_r2s = _r2_for_pred(valid_vecs, obs_vec)

            null_r2_rows = Parallel(n_jobs=n_cores)(
                delayed(_r2_for_pred)(valid_vecs, nv) for nv in cm_null_vecs[cm_name]
            )
            null_r2_arr = np.array(null_r2_rows)

            for i, (perp, r2) in enumerate(zip(valid_perps, obs_r2s)):
                p_perm = float(np.nanmean(null_r2_arr[:, i] >= r2)) if np.isfinite(r2) else np.nan
                records.append({
                    'subject': perp, 'info': atom_name,
                    'predictor': f'CM_{cm_name}',
                    'r2': r2, 'p_perm': p_perm,
                })

    df = pd.DataFrame(records)
    out_path = f'{out_dir}/matrix_r2.csv'
    df.to_csv(out_path, index=False)
    print(f'saved {out_path} ({len(df):,} rows)')


_group_csv_preds = {
    'topology_global': lambda k: k.startswith('scalar_'),
    'community': lambda k: k.startswith('community_'),
}


def _append_group_csvs(results_list):
    for group_name, pred in _group_csv_preds.items():
        dfs = [df for res in results_list for k, df in res.items() if pred(k)]
        if not dfs:
            continue
        out_path = f'{out_dir}/{group_name}.csv'
        write_header = not os.path.exists(out_path)
        pd.concat(dfs, ignore_index=True).to_csv(out_path, mode='a', header=write_header, index=False)


if __name__ == '__main__':
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    for _g in _group_csv_preds:
        _p = f'{out_dir}/{_g}.csv'
        if os.path.exists(_p):
            os.remove(_p)

    yeo_int = load_yeo_labels()

    # Phase 1: group community detection (serial, all cores per atom)
    group_data = {}
    for atom_name, atom_indices in atoms.items():
        print(f'\n--- {atom_name} (group) ---')
        _, atom_indices_out, group_ci = _group_atom_pipeline(atom_name, atom_indices)
        group_data[atom_name] = (atom_indices_out, group_ci)
        gc.collect()

    # Phase 2a: scalar topology across all tasks in parallel
    print('\n--- per-subject first-pass metrics (all atoms) ---')
    first_tasks = [
        (perp, atom_indices, atom_name)
        for atom_name, (atom_indices, _) in group_data.items()
        for perp in subjects
    ]
    first_results = Parallel(n_jobs=n_cores)(
        delayed(process_subject_first)(*t) for t in first_tasks
    )
    _append_group_csvs(first_results)
    del first_results
    gc.collect()

    # Phase 2b: community measures serially per atom so each uses all cores
    print('\n--- per-subject second-pass metrics (all atoms) ---')
    for atom_name, (atom_indices, group_ci) in group_data.items():
        atom_second = []
        for perp in subjects:
            res = process_subject_second(
                perp, atom_indices, atom_name, group_ci, yeo_int, n_jobs=n_cores
            )
            atom_second.append(res)
        _append_group_csvs(atom_second)
        del atom_second
        gc.collect()

    del group_data
    gc.collect()

    compute_direct_indirect_analysis()
    compute_matrix_r2_tests()
    print('\ndone.')

print('Your time has come, run to the hills')
print('Live your life as though every act were to become a universal law')
