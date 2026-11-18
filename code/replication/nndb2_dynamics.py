import os
import sys
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import functools
print = functools.partial(print, flush=True)
import itertools

import numpy as np
from pathlib import Path
from scipy.io import loadmat
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score, pairwise_distances_argmin
from joblib import Parallel, delayed

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
sys.path.insert(0, root) # dynamic_utils and the pyNuMIT package live at the repo root
sys.path.insert(0, here) # replication misc.py must shadow the one at the root

import misc
from dynamic_utils import entropy_rate, normalise

n_cores = 50

# --- config ---
k_min = 2
k_max = 14
n_init = 50 # k-means restarts per partition, keep the best separation
n_partitions = 10 # independent repeats; pick the one sharing most AMI with the rest
kmeans_batch = 10000
red_group = 'Oneness' # redundancy atom
syn_group = 'Cooperation' # synergy atom

base_dir = Path(here) / 'nndb2_results' # anchored to replication/, not the launch dir
results_base = base_dir / 'main_dynamic' / misc.parcellation
main_base = base_dir / 'main' / misc.parcellation
out_dir = base_dir / 'dynamics' / misc.parcellation
sel_dir = out_dir / 'gradient_selection'

subjects = list(misc.subjects)
n_subjects = len(subjects)
n_regions = misc.n_regions
all_groups = list(misc.atoms.keys())
out_dir.mkdir(parents=True, exist_ok=True)
sel_dir.mkdir(parents=True, exist_ok=True)

def load_subject_data(sub_id, groups):
    path = results_base / f'sub-{sub_id}' / 'phiid_regional.mat'
    mat = loadmat(str(path))
    return np.stack([mat[g] for g in groups], axis=-1)


def _er_worker(ts):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        # Fs is 1/TR, 1/1.5 Hz here against the original dataset's 1 Hz. Note that
        # it does not rescale anything: StateSpaceEntropyRate only reads Fs to pick a
        # downsampling factor above 200 Hz and to set band edges, and neither applies
        # here (band is None), so the broadband value is nats per sample either way.
        # The two datasets are therefore NOT on a common per-second scale, and only
        # the topography is comparable between them, not the absolute magnitudes
        return entropy_rate(ts, Fs=misc.sfreq)

def compute_entropy_rates(data):
    n_regions, _, n_atoms = data.shape
    ts_list = [data[r, :, a] for r in range(n_regions) for a in range(n_atoms)]
    ers = Parallel(n_jobs=n_cores)(delayed(_er_worker)(ts) for ts in ts_list)
    return np.array(ers).reshape(n_regions, n_atoms)

# --- entropy rates (all atom groups) ---
er_path = out_dir / 'entropy_rates.npy'
if not er_path.exists():
    print(f'Computing entropy rates over {len(all_groups)} atom groups...')
    all_er = []
    for i, sub_id in enumerate(subjects):
        print(f'  sub-{sub_id} ({i + 1}/{n_subjects})')
        data = normalise(load_subject_data(sub_id, groups=all_groups))
        all_er.append(compute_entropy_rates(data))
    np.save(er_path, np.stack(all_er))
    print(f'Saved entropy rates -> {er_path}')
else:
    print('Entropy rates already exist, skipping.')


# --- syn-red grad
def ranks(x):
    # per-timepoint rank over regions (axis 0), matching get_synred_grad
    return np.argsort(np.argsort(x, axis=0), axis=0) + 1

def synred_grad(syn, red):
    return ranks(syn) - ranks(red)

def grad_corr(grad, ref):
    # pearson of each timepoint's gradient against a static reference map
    g = grad - grad.mean(0, keepdims=True)
    r = ref - ref.mean()
    return (g * r[:, None]).sum(0) / (np.sqrt((g ** 2).sum(0)) * np.sqrt((r ** 2).sum()))

print('Loading static reference gradients...')
indiv_ref = np.load(main_base / 'indiv_syn_red_grad.npy') # (n_regions, n_subjects)
group_ref = np.load(main_base / 'group_syn_red_grad.npy') # (n_regions,)

if indiv_ref.shape[1] != n_subjects:
    raise ValueError(f'indiv_syn_red_grad has {indiv_ref.shape[1]} columns for '
                     f'{n_subjects} subjects; run nndb2_phiid_s.ipynb first')

# --- per-timepoint gradients + correlations
run_clustering = not (sel_dir / 'k_list.npy').exists()
print('Computing per-timepoint gradients and correlations...')
X_list, lengths = [], []
for i, sub_id in enumerate(subjects):
    sub_out = out_dir / f'sub-{sub_id}'
    sub_out.mkdir(parents=True, exist_ok=True)
    grad_path = sub_out / 'syn_red_grad.npy'
    if grad_path.exists():
        grad = np.load(grad_path)
    else:
        d = load_subject_data(sub_id, [red_group, syn_group])
        grad = synred_grad(d[:, :, 1], d[:, :, 0]).astype(np.int16)
        np.save(grad_path, grad)
    gf = grad.astype(np.float64)
    corr = np.column_stack([grad_corr(gf, indiv_ref[:, i]), grad_corr(gf, group_ref)])
    np.save(sub_out / 'grad_corr.npy', corr.astype(np.float32))
    if run_clustering:
        X_list.append(grad.T.astype(np.float32))
        lengths.append(grad.shape[1])

if not run_clustering:
    print('Clustering already on disk (gradient_selection/), skipping the sweep.')
    print('\nCheck it over bruvaaaaaa')
    sys.exit(0)

X = np.concatenate(X_list, axis=0)
lengths = np.array(lengths, dtype=int)
del X_list
# pearson-correlation metric: z-score each timepoint so euclidean k-means clusters by correlation
X -= X.mean(1, keepdims=True)
sd = X.std(1, keepdims=True)
sd[sd == 0] = 1.0
X /= sd
n_total = X.shape[0]
total_ss = float(((X - X.mean(0)) ** 2).sum())
print(f'Gradient matrix: {X.shape} ({len(subjects)} subjects)')

starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
ends = np.cumsum(lengths)


def kmeans_bic(labels, centers, inertia, n_samples):
    """
    Clusters are modelled as isotropic Gaussians sharing one variance estimated
    from the pooled within-cluster sum of squares
    -2 logL + p ln(N) with p = m * (d + 1) free parameters
    (m centres in d dimensions plus the shared variance)

    Parameters
    ----------
    labels : array (n_samples,) - cluster assignment per sample
    centers : array (m, d) - cluster centres
    inertia : float - total within-cluster sum of squared distances (model.inertia_)
    n_samples : int
    """
    m, d = centers.shape
    n = np.bincount(labels, minlength=m).astype(float)
    var = inertia / (d * (n_samples - m))
    var = max(var, 1e-12)
    nz = n > 0
    log_lik = np.sum(n[nz] * np.log(n[nz])
                     - n[nz] * np.log(n_samples)
                     - 0.5 * n[nz] * d * np.log(2 * np.pi * var)
                     - 0.5 * (n[nz] - 1) * d)
    n_params = m * (d + 1)
    return -2.0 * log_lik + n_params * np.log(n_samples)


def _km_worker(data, K, seed):
    # data is passed explicitly so joblib memmaps it once and shares it read-only
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        km = MiniBatchKMeans(n_clusters=K, random_state=seed, batch_size=kmeans_batch,
                             n_init=1, max_iter=100).fit(data)
    return km.inertia_, km.cluster_centers_.astype(np.float32)


# --- how many distinct gradients? correlation k-means ---
k_list = list(range(k_min, k_max + 1))
ve = np.full(len(k_list), np.nan)
bic = np.full(len(k_list), np.nan)
mean_ami = np.full(len(k_list), np.nan)
max_centroid_corr = np.full(len(k_list), np.nan)
inertia = np.full(len(k_list), np.nan)

for i, K in enumerate(k_list):
    print(f'K={K}: {n_partitions} partitions x {n_init} inits...')
    tasks = [(p, p * 10000 + j) for p in range(n_partitions) for j in range(n_init)]
    res = Parallel(n_jobs=n_cores, max_nbytes='1M')(
        delayed(_km_worker)(X, K, s) for _, s in tasks)

    # best separation (lowest inertia) within each independent partition
    part_inertia = np.full(n_partitions, np.inf)
    part_centers = [None] * n_partitions
    for (p, _), (inert, cent) in zip(tasks, res):
        if inert < part_inertia[p]:
            part_inertia[p] = inert
            part_centers[p] = cent
    part_labels = [pairwise_distances_argmin(X, c) for c in part_centers]

    # representative partition: the one sharing the most AMI with all others
    ami = np.zeros((n_partitions, n_partitions))
    for a, b in itertools.combinations(range(n_partitions), 2):
        v = adjusted_mutual_info_score(part_labels[a], part_labels[b])
        ami[a, b] = ami[b, a] = v
    rep = int(ami.sum(1).argmax())
    mean_ami[i] = ami[rep].sum() / (n_partitions - 1)
    labels = part_labels[rep]
    cent = part_centers[rep]

    diff = X - cent[labels]
    inertia[i] = float(np.einsum('ij,ij->', diff, diff))
    ve[i] = 1.0 - inertia[i] / total_ss
    bic[i] = float(kmeans_bic(labels, cent, inertia[i], n_total))
    cc = np.corrcoef(cent)
    np.fill_diagonal(cc, 0.0)
    max_centroid_corr[i] = float(np.abs(cc).max())

    occ = np.zeros((n_subjects, K))
    for si in range(n_subjects):
        lab = labels[starts[si]:ends[si]]
        occ[si] = np.bincount(lab, minlength=K) / lab.size
    coverage = (occ > 0).mean(0)

    k_dir = out_dir / f'K{K}'
    k_dir.mkdir(parents=True, exist_ok=True)
    np.save(k_dir / 'centroids.npy', cent)
    np.save(k_dir / 'labels.npy', labels.astype(np.int8))
    np.savez(k_dir / 'qc.npz', occupancy=occ, coverage=coverage, group_occ=occ.mean(0),
             centroid_corr=cc, max_centroid_corr=max_centroid_corr[i], ve=ve[i],
             bic=bic[i], mean_ami=mean_ami[i], ami=ami, inertia=inertia[i])
    dve = 0.0 if i == 0 else ve[i] - ve[i - 1]
    print(f'  K={K:2d}  VE={ve[i]:.3f}  dVE={dve:+.3f}  BIC={bic[i]:.4e}  AMI={mean_ami[i]:.3f}  '
          f'max|cent r|={max_centroid_corr[i]:.2f}  coverage>={coverage.min() * 100:.0f}%')

np.save(sel_dir / 'k_list.npy', np.array(k_list))
np.save(sel_dir / 've.npy', ve)
np.save(sel_dir / 'bic.npy', bic)
np.save(sel_dir / 'mean_ami.npy', mean_ami)
np.save(sel_dir / 'max_centroid_corr.npy', max_centroid_corr)
np.save(sel_dir / 'inertia.npy', inertia)
print('\nCheck it over bruvaaaaaa')
