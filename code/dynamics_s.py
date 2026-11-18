# %%
# libraries
import os
import sys
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import warnings
import cmasher as cmr
import palettable as pl
import pingouin as pg
from scipy.stats import spearmanr, zscore
from scipy.stats import zscore
import statsmodels.stats.multitest as multi
import warnings
from tqdm import tqdm
from scipy.io import loadmat
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.font_manager as fm
font_path = '/home/george/Research/fonts/Instrument_Sans/static/InstrumentSans-Regular.ttf'
available_fonts = [f.name for f in fm.fontManager.ttflist]
if 'Instrument Sans' not in available_fonts:
    fm.fontManager.addfont(font_path)
    plt.rcParams.update({'font.family': 'Instrument Sans'})
else:
    plt.rcParams['font.family'] = 'Instrument Sans'
plt.rcParams['font.family'] = 'Instrument Sans'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 800
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = "cm"
rc_fontsize = 15
plt.rcParams.update({'font.size': rc_fontsize})

import network_utils as nu
import misc

parcellation = misc.parcellation
n_regions = misc.n_regions

MRI_path = '/home/george/Research/fMRI_general'
atlas_info = pd.read_csv(f'{MRI_path}/atlases/schaefer_extend_atlases/{parcellation}.csv')

main_dir = f'results/main'
dynamics_dir = f'results/dynamics'

plots_dir = f'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

import sys
sys.path.append(f'{MRI_path}')
from cortexcanvas import (plot_cortex, 
                          plot_wholebrain_row, plot_wholebrain_row_2, 
                          plot_full,
                          plot_wholebrain_and_flat)

modes = misc.atom_order  # the 10 modes, in a sensible order

from palettable.scientific.diverging import Berlin_20
berlin_cmap = Berlin_20.mpl_colormap

recompute = False  # set True to force any cached analysis below to recompute


# %%
# correlate the trajectories of the 10 information-dynamics modes within one region of one subject
example_sub = 1
example_region = 0

main_dynamic_dir = f'results/main_dynamic/{parcellation}'
mat = loadmat(f'{main_dynamic_dir}/sub-{example_sub}/phiid_regional.mat')

# (n_time, 10): time course of each mode in this region
traj = np.stack([mat[m][example_region] for m in modes], axis=1)

print(f'Subject {example_sub} has {traj.shape[0]} time points.')

# mode x mode correlation across time
corr = np.corrcoef(traj.T)
from palettable.scientific.diverging import Vik_3
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(corr, cmap=Vik_3.mpl_colormap, vmin=-1, vmax=1)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                    label='r')
cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(4))
ax.set_xticks(ticks=np.arange(len(misc.atom_order)))
ax.set_xticklabels(labels=[misc.atoms_names[atom] for atom in misc.atom_order], rotation=90)
ax.set_yticks(ticks=np.arange(len(misc.atom_order)))
ax.set_yticklabels(labels=[misc.atoms_names[atom] for atom in misc.atom_order])
ax.set_xticks(np.arange(-0.5, len(misc.atom_order), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(misc.atom_order), 1), minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()


# %%
# plot histogram
import glob
accent = '#6B1E4E'

# every timepoint's correlation to the static gradient, collapsed across all subjects
files = sorted(glob.glob(f'{dynamics_dir}/{parcellation}/sub-*/grad_corr.npy'))
corr_all = np.concatenate([np.load(f) for f in files], axis=0)

for col, label, fname in [(0, 'Sim. to static', 'corr_hist_individual'),
                          (1, 'Sim. to static', 'corr_hist_group')]:
    vals = corr_all[:, col]
    print(f'{label}: r = {np.mean(vals):.3f} +/- {np.std(vals):.3f}')
    fig, ax = plt.subplots(figsize=(2.5, 1.4))
    ax.hist(vals, bins=120, range=(-0.4, 0.6), color='lightgrey')
    ax.axvline(0, color='black', lw=0.6, ls='-')
    ax.set_xlabel(label)
    ax.set_yticks([])
    ax.set_xlim(-0.4, 0.6)
    ax.set_xlim(-max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1])), max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1])))
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    sns.despine(left=True)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/dyn_synred_{fname}.png', dpi=800, bbox_inches='tight')
    plt.show()


# %%
# plot quality metrics
accent = '#6B1E4E'
sel = f'{dynamics_dir}/{parcellation}/gradient_selection'
k_list = np.load(f'{sel}/k_list.npy')
ve = np.load(f'{sel}/ve.npy')
bic = np.load(f'{sel}/bic.npy')
max_cc = np.load(f'{sel}/max_centroid_corr.npy')

ami_dist, min_cov = [], []
for K in k_list:
    q = np.load(f'{dynamics_dir}/{parcellation}/K{K}/qc.npz')
    a = q['ami']
    ami_dist.append(a[np.triu_indices_from(a, k=1)])
    min_cov.append(q['coverage'].min())
min_cov = np.array(min_cov)
dve = np.r_[np.nan, np.diff(ve)]  # gain in variance explained per +1 state

fig, ax = plt.subplots(2, 3, figsize=(8.5, 4))

a = ax[0, 0]
a.plot(k_list, ve, '-o', color=accent)
a.set(ylabel='Var. expl. (VE)')

a = ax[0, 1]
a.plot(k_list, dve, '-o', color=accent)
a.set(ylabel=r'$\Delta$ VE')

a = ax[0, 2]
a.plot(k_list, bic, '-o', color=accent)
a.set(ylabel='BIC')

a = ax[1, 0]
a.boxplot(ami_dist, positions=k_list, showmeans=True, showfliers=False)
a.set(xlabel='K', ylabel='Reproduc. (AMI)', ylim=(0, 1))

a = ax[1, 1]
a.plot(k_list, max_cc, '-o', color=accent)
a.set(xlabel='K', ylabel=r'Max state corr.', ylim=(-0.05, 1.05))

a = ax[1, 2]
a.plot(k_list, min_cov, '-o', color=accent)
a.set(xlabel='K', ylabel='Min coverage', ylim=(0, 1.05))

for a in ax.flat:
    sns.despine(ax=a)
    a.yaxis.set_major_locator(plt.MaxNLocator(2))
    a.set_xticks([5, 10])
    a.set_xticklabels([5, 10])
fig.align_ylabels(ax)
fig.tight_layout()
plt.savefig(f'{plots_dir}/dyn_synred_K_QC.png', dpi=800, bbox_inches='tight')
plt.show()


# %%
# very low VE lol, almost non-recuring spatial structure...


# %%
chosen_K = 3

# NB: scale of clusters diff cus z-scored (per timepoint) gradients,
# did it so that clustering works on pearson correlation distance, not euclidean distance
# and thus wont make seperate clusters for the same spatial pattern but different magnitudes
# 0 should still mean balanced syn-red,
# since mean of a rank - rank (two permutations of the same len) = 0

kdir = f'{dynamics_dir}/{parcellation}/K{chosen_K}'
labels = np.load(f'{kdir}/labels.npy') # cluster id per concatenated timepoint
group_occ = np.load(f'{kdir}/qc.npz')['group_occ'] # mean occupancy per state

# but lets put back into rank space for plotting
sums = np.zeros((chosen_K, n_regions))
counts = np.zeros(chosen_K)
pos = 0
for s in misc.subjects:
    g = np.load(f'{dynamics_dir}/{parcellation}/sub-{s}/syn_red_grad.npy')
    T = g.shape[1]
    lab = labels[pos:pos + T]
    pos += T
    for k in range(chosen_K):
        m = lab == k
        if m.any():
            sums[k] += g[:, m].sum(1)
            counts[k] += m.sum()
centroids = sums / counts[:, None] # (K, n_regions) in rank-difference units

order = np.argsort(group_occ)[::-1] # most-occupied state first
lim = 150

for rank, s in enumerate(order, start=1):
    print(f'State {rank}: {group_occ[s] * 100:.1f}% occupancy')
    fig = plot_wholebrain_row(centroids[s], parcellation, cmap=misc.sr_full_cmap,
                              vmin=-lim, vmax=lim, brightness=0.4)
    fig.savefig(f'{plots_dir}/K{chosen_K}_synred_state{rank}.png', dpi=700, bbox_inches='tight')
    plt.show()


# %%
fig, ax = plt.subplots(1, 1, figsize=(2, .2))
cbar = plt.colorbar(
    cm.ScalarMappable(cmap=misc.sr_full_cmap), 
    cax=ax, 
    orientation='horizontal'
)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels([r'$\leq$-150', '', r'$\geq$150'])
cbar.ax.tick_params(labelsize=rc_fontsize)
# label
cbar.set_label('Rank', fontsize=rc_fontsize, labelpad=-10)
secax = cbar.ax.secondary_xaxis('top')
secax.set_xticks([0, 1])
secax.set_xticklabels([r'$R \rightarrow R$', r'$S \rightarrow S$'])
secax.tick_params(axis='x', pad=.5)
secax.tick_params(labelsize=rc_fontsize)
plt.savefig(f'{plots_dir}/k{K}_synred_cbar.png', bbox_inches='tight', transparent=True)


# %%
shared_cbar = True

# plot dynamics grid
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import loadmat
from scipy.stats import zscore

perp = misc.subjects[0]
mat = loadmat(f'results/main_dynamic/{parcellation}/sub-{perp}/phiid_regional.mat')

def plot_dynamics_grid(mat, atoms_to_plot, shared_cbar=True, shared_cmap=False, 
                       cmap=None, draw_tpoint=False, tpoint=None):

    n_samples = mat[atoms_to_plot[0]].shape[1]
    atom_data = {atom: zscore(mat[atom].astype(np.float64), axis=1) for atom in atoms_to_plot}
    if shared_cmap:
        atom_cmaps = {
            atom: cmap for atom in atoms_to_plot
        }
    else:
        atom_cmaps = {
            atom: LinearSegmentedColormap.from_list(atom, ['white', 'grey', misc.atoms_colors[atom]])
            for atom in atoms_to_plot
        }

    if shared_cbar:
        global_vmin = min(np.percentile(d, 5) for d in atom_data.values())
        global_vmax = max(np.percentile(d, 95) for d in atom_data.values())
        atom_vlims = {atom: (global_vmin, global_vmax) for atom in atoms_to_plot}
    else:
        atom_vlims = {
            atom: (np.percentile(atom_data[atom], 5), np.percentile(atom_data[atom], 95))
            for atom in atoms_to_plot
        }

    n_atoms_plot = len(atoms_to_plot)
    height_ratios = [3] * n_atoms_plot
    fig, axes = plt.subplots(n_atoms_plot, 1, figsize=(11 * 0.8, n_atoms_plot * 1),
                             sharex=True,
                             gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.2})


    inset_locs = [
        [0.025, -0.05, 0.15, 0.1],
        [0.225, -0.05, 0.15, 0.1],
        [0.425, -0.05, 0.15, 0.1],
        [0.625, -0.05, 0.15, 0.1],
        [0.825, -0.05, 0.15, 0.1],
    ]
    zoom_indices = [
        [0, 20],
        [int(n_samples * 0.2), int(n_samples * 0.2) + 20],
        [int(n_samples * 0.4), int(n_samples * 0.4) + 20],
        [int(n_samples * 0.6), int(n_samples * 0.6) + 20],
        [int(n_samples * 0.8), int(n_samples * 0.8) + 20],
    ]

    images = {}
    for i, atom in enumerate(atoms_to_plot):
        data = atom_data[atom]
        cmap = atom_cmaps[atom]
        vmin, vmax = atom_vlims[atom]
        ax = axes[i]
        is_last = (i == n_atoms_plot - 1)

        images[atom] = ax.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        sns.despine(ax=ax, left=True, bottom=True, right=True, top=True)
        ax.set_yticks([0, 999])
        ax.set_yticklabels(['1', '1000'])
        ax.tick_params(axis='y', length=0)
        ax.set_xticks([])

        for j, zoom_id in enumerate(zoom_indices):
            if is_last and j == 0:
                axins = ax.inset_axes(
                    inset_locs[j],
                    xlim=(zoom_id[0], zoom_id[1]),
                    ylim=(0, data.shape[0]),
                    yticklabels=[], yticks=[],
                )
                axins.set_xlabel('Time (s)',  labelpad=0)
            else:
                axins = ax.inset_axes(
                    inset_locs[j],
                    xlim=(zoom_id[0], zoom_id[1]),
                    ylim=(0, data.shape[0]),
                    xticklabels=[], yticklabels=[], xticks=[], yticks=[],
                )
            axins.imshow(data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
            ax.indicate_inset_zoom(axins, edgecolor='black')

    axes[n_atoms_plot // 2].set_ylabel('Region')
    fig.align_ylabels(axes)
    plt.tight_layout(rect=[0, 0, 0.9, 1], h_pad=1)

    for i, atom in enumerate(atoms_to_plot):
        vmin, vmax = atom_vlims[atom]
        pos = axes[i].get_position()
        cbar_ax = fig.add_axes([0.91, pos.y0 + pos.height * 0.15, 0.01, pos.height * 0.7])
        cb = fig.colorbar(images[atom], cax=cbar_ax)
        cb.set_ticks([np.round(vmin, 1), np.round(vmax, 1)])
        cbar_ax.set_ylabel(misc.atoms_names[atom], rotation=90, labelpad=rc_fontsize, va='center')

    if draw_tpoint and tpoint is not None:
        for ax in axes:
            ax.axvline(tpoint, color='black', linestyle='-', linewidth=1)

    return fig, axes


from palettable.scientific.diverging import Vik_3 
#fig, axes = plot_dynamics_grid(mat, misc.atoms_nr, shared_cbar=shared_cbar)
#plt.savefig(f'{plots_dir}/dynamics_grid.png', bbox_inches='tight')
fig, axes = plot_dynamics_grid(mat, misc.atoms_pol, shared_cbar=shared_cbar,
                                 draw_tpoint=True, tpoint=2125)
plt.savefig(f'{plots_dir}/dynamics_grid_pol.png', bbox_inches='tight')


# %%
# plot a histogram for entropy rates 

entropy_rates = np.load(f'results/dynamics/{parcellation}/entropy_rates.npy')  # (n_subjects, n_regions, groups)

fig, ax = plt.subplots(figsize=(3, 1))
ax.hist(entropy_rates.flatten(), bins=100, color='grey',
        cumulative=True, density=True, histtype='step', linewidth=2)
ax.set_xlabel('Entropy Rate')
ax.set_ylabel('Freq.')
sns.despine(ax=ax)


# %%
entropy_rates = np.load(f'results/dynamics/{parcellation}/entropy_rates.npy')  # (n_subjects, n_regions, groups)
mean_er = np.nanmean(entropy_rates, axis=0)  # (n_regions, groups)
mean_mean_er = np.nanmean(mean_er, axis=-1)

print('-' * 100)
print(f'Entropy rates')
print('-' * 100)
misc.print_topography_extremes(mean_mean_er, n=50, parc=parcellation, expand=True)


# %%
# plot entropy rate topographic maps

entropy_rates = np.load(f'results/dynamics/{parcellation}/entropy_rates.npy')  # (n_subjects, n_regions, groups)
mean_er = np.nanmean(entropy_rates, axis=0)  # (n_regions, groups)

for a, atom in enumerate(misc.atoms_nr):
    values = mean_er[:, list(misc.atoms.keys()).index(atom)]
    vmin = np.nanpercentile(values, 5)
    vmax = np.nanpercentile(values, 95)

    fig = plot_wholebrain_row_2(values, parcellation, brightness=0.4,
                                cmap=berlin_cmap, vmin=vmin, vmax=vmax)
    fig.savefig(f'{plots_dir}/entropy_rate_{atom}.png', bbox_inches='tight')
    #plt.close(fig)

    dp = 2 if f'{vmin:.2f}' != f'{vmax:.2f}' else 3
    fig, ax = plt.subplots(1, 1, figsize=(0.1, 0.6))
    cbar = plt.colorbar(cm.ScalarMappable(cmap=berlin_cmap), cax=ax, orientation='vertical')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([f'{vmin:.{dp}f}', f'{vmax:.{dp}f}'], fontsize=13)
    fig.savefig(f'{plots_dir}/entropy_rate_{atom}_cbar.svg', bbox_inches='tight', transparent=True)
    #plt.close(fig)

# low dimensional topography of temporal diversity across atoms

from sklearn.decomposition import PCA
n_components = 10
pca = PCA(n_components=n_components)
pca.fit(mean_er.T)
explained_var = pca.explained_variance_ratio_
components = pca.components_

# plot explained variance
fig, ax = plt.subplots(1, 1, figsize=(2, 1))
ax.plot(range(1, n_components + 1), explained_var, marker='.', linestyle='-', color='grey')
ax.set_xlabel('Component', labelpad=0)
ax.set_ylabel(r'$\frac{\lambda_i}{\sum \lambda}$', rotation=0, labelpad=10)
ax.xaxis.set_major_locator(plt.MaxNLocator(2))
ax.set_yticks([0, 1])
ax.set_ylim(-0.1, 1.05)
sns.despine(ax=ax)
fig.savefig(f'{plots_dir}/ER_pca_explained_variance.png', bbox_inches='tight')

#plot topography of first components
for i, component in enumerate(components[:1]):
    vmin = np.nanpercentile(component, 5)
    vmax = np.nanpercentile(component, 95)
    fig = plot_full(component, parcellation, brightness=0.4,
                                cmap=berlin_cmap, vmin=vmin, vmax=vmax)
    fig.savefig(f'{plots_dir}/ER_pca_component_{i+1}.png', bbox_inches='tight', transparent=True)

    # plot cbar
    dp = 2 if f'{vmin:.2f}' != f'{vmax:.2f}' else 3
    fig, ax = plt.subplots(1, 1, figsize=(0.1, 0.6))
    cbar = plt.colorbar(cm.ScalarMappable(cmap=berlin_cmap), cax=ax, orientation='vertical')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([f'{vmin:.{dp}f}', f'{vmax:.{dp}f}'], fontsize=13)
    fig.savefig(f'{plots_dir}/ER_pca_component_{i+1}_cbar.svg', bbox_inches='tight', transparent=True)

# just plot the mean across atoms
mean_across_atoms = np.nanmean(mean_er, axis=1)
vmin = np.nanpercentile(mean_across_atoms, 5)
vmax = np.nanpercentile(mean_across_atoms, 95)
fig = plot_full(mean_across_atoms, parcellation, brightness=0.4,
                            cmap=berlin_cmap, vmin=vmin, vmax=vmax)
fig.savefig(f'{plots_dir}/ER_mean_across_atoms.png', bbox_inches='tight', transparent=True)

dp = 2 if f'{vmin:.2f}' != f'{vmax:.2f}' else 3
fig, ax = plt.subplots(1, 1, figsize=(0.8, 0.1))
cbar = plt.colorbar(cm.ScalarMappable(cmap=berlin_cmap), cax=ax, orientation='horizontal')
cbar.set_ticks([0, 1])
cbar.set_ticklabels([f'{vmin:.{dp}f}', f'{vmax:.{dp}f}'], fontsize=13)
fig.savefig(f'{plots_dir}/ER_mean_across_atoms_cbar.svg', bbox_inches='tight', transparent=True)


# %%
# winner-takes-all mode state sequence for every participant
from dynamic_utils import normalise, state_dynamics, generate_tpm

n_states = len(modes)
winner_dir = f'results/dynamics_winner/{parcellation}'
os.makedirs(winner_dir, exist_ok=True)

pref = np.array([1 if m in misc.atoms_nr else 0 for m in modes])

winner_dynamics = {}
for sub in tqdm(misc.subjects):
    sub_out = f'{winner_dir}/sub-{sub}'
    npz_path = f'{sub_out}/dynamics.npz'

    if os.path.exists(npz_path):  # load cached result instead of recomputing
        winner_dynamics[sub] = dict(np.load(npz_path))
        continue

    mat = loadmat(f'{main_dynamic_dir}/sub-{sub}/phiid_regional.mat')
    data = np.stack([mat[m] for m in modes], axis=-1)  # (n_regions, n_time, n_modes)

    # z-score each region/mode time course, then average across regions
    avg = np.nanmean(normalise(data, axis=1), axis=0)  # (n_time, n_modes)

    # winner-takes-all per time point; ties broken in favour of modes in atoms_nr
    is_max = avg == avg.max(axis=1, keepdims=True)
    state_seq = (is_max * (1 + pref)).argmax(axis=1) + 1  # 1-indexed states

    f_occ, dt, vt, ar = state_dynamics(state_seq, n_states, TR=misc.tau)
    tpm = generate_tpm(state_seq, n_states)

    result = dict(
        state_seq=state_seq,
        tpm=tpm,
        fo=np.array([f_occ[s] for s in range(1, n_states + 1)]),
        dt=np.array([dt[s] for s in range(1, n_states + 1)]),
        vt=np.array([vt[s] for s in range(1, n_states + 1)]),
        ar=np.array([ar[s] for s in range(1, n_states + 1)]),
    )
    os.makedirs(sub_out, exist_ok=True)
    np.savez(npz_path, **result)
    winner_dynamics[sub] = result


# %%
# example viterbi path

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

state_labels = [misc.atoms_names[m] for m in modes]
state_colors = [misc.atoms_colors[m] for m in modes]
state_cmap = ListedColormap(state_colors)
norm = BoundaryNorm(np.arange(0.5, n_states + 1.5), state_cmap.N)

seq = winner_dynamics[example_sub]['state_seq']
zoom = slice(0, 60)

fig, ax = plt.subplots(figsize=(4*1.25, .4*1.25))
ax.imshow(seq[None, :], aspect='auto', cmap=state_cmap, norm=norm,
          interpolation='nearest', extent=[0, len(seq), 0, 1])
ax.set_yticks([])
ax.xaxis.set_major_locator(plt.MaxNLocator(2))

# zoomed inset drawn below the full path, with connector lines to the zoom region
axins = ax.inset_axes([0.1, -1.25, 0.3, 0.6])
axins.imshow(seq[None, :], aspect='auto', cmap=state_cmap, norm=norm,
             interpolation='nearest', extent=[0, len(seq), 0, 1])
axins.set_xlim(zoom.start, zoom.stop)
#axins.set_xlabel('Time (s)', labelpad=0)
axins.set_ylim(0, 1)
axins.set_yticks([])
ax.indicate_inset_zoom(axins, edgecolor='black', linestyle='-', linewidth=1, alpha=1)

# handles = [Patch(facecolor=c, label=l) for c, l in zip(state_colors, state_labels)]
# fig.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
# plt.tight_layout()
# plt.show()

fig.savefig(f'{plots_dir}/winner_example_path.png', bbox_inches='tight')


# %%
# state dynamics statistics
import itertools

tr = 1  # repetition time in seconds
metrics = [('ar', 'Freq. (/s)'), ('dt', 'Dwell (s)'), ('fo', 'FO (%)')]

# long-format stats across participants (one row per subject x mode); all metrics
# are film-length independent (fo is a proportion, dt a mean duration, ar visits/s).
rows = []
for sub in misc.subjects:
    d = winner_dynamics[sub]
    for i, m in enumerate(modes):
        rows.append({'subject': sub, 'atom': m,
                     'fo': d['fo'][i], 'dt': d['dt'][i],
                     'vt': d['vt'][i], 'ar': d['ar'][i] / tr})
stats_df = pd.DataFrame(rows)

np.random.seed(42)
bp_kw = dict(showcaps=False, showfliers=False, patch_artist=True, widths=0.5,
             boxprops=dict(facecolor='none', edgecolor='black', linewidth=1, zorder=10),
             medianprops=dict(color='black', linewidth=1, zorder=11),
             whiskerprops=dict(color='black', linewidth=1, zorder=11))

fig, axs = plt.subplots(len(metrics), 1, figsize=(3.8*1.2, (1.35 * len(metrics))*1.25), sharex=True)
for ax, (metric, label) in zip(axs, metrics):
    for i, m in enumerate(misc.atom_order):
        vals = stats_df[stats_df['atom'] == m][metric].values
        if metric == 'fo':
            vals = vals * 100
        ax.scatter(np.full(len(vals), i) + np.random.normal(0, 0.06, len(vals)), vals,
                   color=misc.atoms_colors[m], alpha=0.6, s=12, zorder=5)
        ax.boxplot(vals, positions=[i], **bp_kw)
    ax.set_ylabel(label)
    ax.set_xlim(-0.5, len(misc.atom_order) - 0.5)
    sns.despine(ax=ax)

axs[-1].set_xticks(range(len(misc.atom_order)))
axs[-1].set_xticklabels([misc.atoms_names[m] for m in misc.atom_order], rotation=90)
for tl in axs[-1].get_xticklabels():
    key = [k for k, v in misc.atoms_names.items() if v == tl.get_text()][0]
    tl.set_color(misc.atoms_colors[key])
fig.align_ylabels(axs)
plt.tight_layout()
fig.savefig(f'{plots_dir}/winner_state_dynamics.png', bbox_inches='tight')
plt.show()


# RM-ANOVA (within = mode) + FDR pairwise t-tests per metric (cached)
if not recompute and os.path.exists(f'{winner_dir}/winner_state_stats.pkl'):
    R = pickle.load(open(f'{winner_dir}/winner_state_stats.pkl', 'rb'))
    anova_stats, pairwise_stats = R['anova_stats'], R['pairwise_stats']
else:
    atom_pairs = list(itertools.combinations(misc.atom_order, 2))
    anova_rows, pairwise_rows = [], []
    for metric, label in metrics:
        df_long = stats_df[['subject', 'atom', metric]].rename(columns={metric: 'value'})
        anova = pg.rm_anova(data=df_long, dv='value', within='atom', subject='subject')
        anova_rows.append({'metric': metric,
                           'F': anova['F'].values[0],
                           'ddof1': anova['ddof1'].values[0],
                           'ddof2': anova['ddof2'].values[0],
                           'p': anova['p_unc'].values[0],
                           'ng2': anova['ng2'].values[0]})

        pair_stat_rows = []
        for atom_a, atom_b in atom_pairs:
            va = stats_df[stats_df['atom'] == atom_a][metric].values
            vb = stats_df[stats_df['atom'] == atom_b][metric].values
            try:
                t, p, cil, ciu, d, bf, dof, power = misc.aug_t_test(va, vb, paired=True)
            except Exception:  # zero-variance pair (e.g. two never-occupied states)
                t = p = cil = ciu = d = bf = dof = power = np.nan
            pair_stat_rows.append({'metric': metric, 'atom_a': atom_a, 'atom_b': atom_b,
                                   't': t, 'p': p, 'd': d, 'dof': dof})

        pair_df = pd.DataFrame(pair_stat_rows)
        pbh = np.full(len(pair_df), np.nan)
        finite = np.isfinite(pair_df['p'].values)
        if finite.any():
            pbh[finite] = multi.multipletests(pair_df['p'].values[finite], method='fdr_bh')[1]
        pair_df['p_bh'] = pbh
        pairwise_rows.append(pair_df)

    anova_stats = pd.DataFrame(anova_rows)
    pairwise_stats = pd.concat(pairwise_rows, ignore_index=True)
    pickle.dump(dict(anova_stats=anova_stats, pairwise_stats=pairwise_stats), open(f'{winner_dir}/winner_state_stats.pkl', 'wb'))

print('RM-ANOVA (within = mode):')
for _, r in anova_stats.iterrows():
    F_str = misc.write_scientific(r['F'])
    p_str = misc.write_scientific(r['p'])
    print(f"  {r['metric']}: F={F_str}, "
          f"p={p_str}, ng2={r['ng2']:.3f}")

print('\nSignificant pairwise comparisons (p_FDR < 0.05):')
sig = pairwise_stats[pairwise_stats['p_bh'] < 0.05]
for _, r in sig.iterrows():
    a, b = misc.atoms_texcommands[r['atom_a']], misc.atoms_texcommands[r['atom_b']]
    print(f"  [{r['metric']}] {a} vs {b}: T={misc.write_scientific(r['t'])}, "
          f"p_FDR={misc.write_scientific(r['p_bh'])}, d={misc.write_scientific(r['d'])}")

# stats for the claims in the text. where the claim is that a mode is the extreme
# one, the supporting number is the largest FDR-corrected p among its contrasts
scales = {'fo': (100, '%', 1), 'dt': (1, 's', 2), 'ar': (1, '/s', 3)}

def describe(metric, atom):
    scale, unit, dp = scales[metric]
    v = stats_df[stats_df['atom'] == atom][metric].values * scale
    return f'{v.mean():.{dp}f}+-{v.std():.{dp}f} {unit}'

def rank_of(metric, atom):
    means = stats_df.groupby('atom')[metric].mean().sort_values(ascending=False)
    return list(means.index).index(atom) + 1

def worst_p(metric, atom, against=None):
    d = pairwise_stats[pairwise_stats['metric'] == metric]
    other = d['atom_b'].where(d['atom_a'] == atom, d['atom_a'])
    keep = (d['atom_a'] == atom) | (d['atom_b'] == atom)
    if against is not None:
        keep &= other.isin(against)
    return d[keep]['p_bh'].max()

print('\nPreservation: highest FO, driven by dwell time rather than appearance rate')
for metric in ['fo', 'dt', 'ar']:
    print(f"  {metric}: {describe(metric, 'Preservation')}, "
          f"rank {rank_of(metric, 'Preservation')}/10, "
          f"max p_FDR={misc.write_scientific(worst_p(metric, 'Preservation'))}")

print('\nCopy and Erasure: large FO, driven by appearance rate')
rest = [m for m in modes if m not in ('Copy', 'Erasure')]
for atom in ['Erasure', 'Copy']:
    print(f"  {atom}: FO rank {rank_of('fo', atom)}/10, "
          f"rate rank {rank_of('ar', atom)}/10, dwell rank {rank_of('dt', atom)}/10")
worst = max(worst_p('ar', 'Copy', rest), worst_p('ar', 'Erasure', rest))
print(f'  rate vs the other eight modes, max p_FDR={misc.write_scientific(worst)}')

print('\nMigration: rarest and most fleeting')
for metric in ['fo', 'dt']:
    print(f"  {metric}: {describe(metric, 'Migration')}, "
          f"rank {rank_of(metric, 'Migration')}/10, "
          f"max p_FDR={misc.write_scientific(worst_p(metric, 'Migration'))}")

print('\nCooperation vs Oneness')
for metric in ['dt', 'ar']:
    print(f"  {metric}: {describe(metric, 'Cooperation')} vs {describe(metric, 'Oneness')}, "
          f"p_FDR={misc.write_scientific(worst_p(metric, 'Cooperation', ['Oneness']))}")


# %%
# Function definition
def tpm_with_self(state_seq, n_states):
    tpm = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            tpm[i, j] = np.sum((state_seq[:-1] == i + 1) & (state_seq[1:] == j + 1))
    row_sums = tpm.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore'):
        tpm = np.where(row_sums > 0, tpm / row_sums, np.nan)
    return tpm

# Data processing
tpms = np.stack([winner_dynamics[sub]['tpm'] for sub in misc.subjects])
mean_tpm = np.nanmean(tpms, axis=0)

tpms_self = np.stack([tpm_with_self(winner_dynamics[sub]['state_seq'], n_states)
                      for sub in misc.subjects])
mean_tpm_self = np.nanmean(tpms_self, axis=0)

# Setup single figure with subplots
cc = cmr.horizon
state_labels = [misc.atoms_names[m] for m in modes]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

matrices = [mean_tpm, mean_tpm_self]
axes = [ax1, ax2]
for ax, mat in zip(axes, matrices):
    im = ax.imshow(mat, cmap=cc)
    
    ax.set_xticks(ticks=np.arange(len(modes)))
    ax.set_xticklabels(labels=state_labels, rotation=90)
    ax.set_yticks(ticks=np.arange(len(modes)))
    ax.set_yticklabels(labels=state_labels)
    
    ax.set_xlabel('$X_{t+1}$')
    ax.set_ylabel('$X_t$')
    
    for tl in ax.get_xticklabels() + ax.get_yticklabels():
        key = [k for k, v in misc.atoms_names.items() if v == tl.get_text()][0]
        tl.set_color(misc.atoms_colors[key])
        
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('$P(X_{t+1} | X_t)$')
    
    for i in range(len(modes) + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5)
        ax.axvline(i - 0.5, color='black', linewidth=0.5)

plt.tight_layout()

# Save data and combined figure
pickle.dump(dict(mean_tpm=mean_tpm), open(f'{winner_dir}/mean_tpm.pkl', 'wb'))
pickle.dump(dict(mean_tpm_self=mean_tpm_self), open(f'{winner_dir}/mean_tpm_self.pkl', 'wb'))
fig.savefig(f'{plots_dir}/straight_tpms.png', bbox_inches='tight')

plt.show()


# %%
# export subject-wise discrete states + legend (python pickle) to share
import pickle

export_dir = f'results/dynamics_winner/{parcellation}/export'
os.makedirs(export_dir, exist_ok=True)

# legend: integer state value -> mode (state s corresponds to modes[s-1])
legend = {s + 1: m for s, m in enumerate(modes)}

# sequences: subject -> 1-indexed state sequence (films differ in length)
sequences = {int(sub): winner_dynamics[sub]['state_seq'].astype(np.int8)
             for sub in misc.subjects}

with open(f'{export_dir}/winner_states.pkl', 'wb') as f:
    pickle.dump({'legend': legend, 'sequences': sequences}, f)

print('legend:', legend)
print('n subjects:', len(sequences), '| example sub-1 length:', len(sequences[1]))


# %%
import pickle
from scipy import stats as sps
from joblib import Parallel, delayed
from matplotlib.colors import to_rgb
import cycle_utils as cu

K = len(modes) # number of information-atom states (10)
tr = misc.sfreq
n_edges = K * K - K
bonf_alpha = 0.05 / n_edges # Bonferroni threshold over all directed off-diagonal edges
n_jobs = max(1, (os.cpu_count() or 2) - 2)

with open(f'results/dynamics_winner/{parcellation}/export/winner_states.pkl', 'rb') as f:
    _w = pickle.load(f)

state_names = [_w['legend'][k] for k in range(1, K + 1)]
node_colors = [to_rgb(misc.atoms_colors[n]) for n in state_names]
subject_ids = [s for s in misc.subjects if int(s) in _w['sequences']]
films = np.array([misc.film_p_dict_r[int(s)] for s in subject_ids])

def to_onehot(seq):
    """1-indexed state sequence -> (n_timepoints, K) one-hot time course."""
    seq = np.asarray(seq).astype(int) - 1
    oh = np.zeros((seq.size, K))
    oh[np.arange(seq.size), seq] = 1.0
    return oh

tcs = [to_onehot(_w['sequences'][int(s)]) for s in subject_ids]

print(f'{len(tcs)} subjects, K={K} states: {state_names}')
print(f'sequence lengths {min(map(len, tcs))}-{max(map(len, tcs))} samples')

opt_restarts = 10 # restarts of the stochastic ordering optimiser
opt_perms = 5 * 10 ** 4   # swap proposals per restart (for the observed ordering)
opt_perms_null = 2 * 10 ** 4  # cheaper optimisation inside each permutation

def asymmetry(fo_density):
    """Per-subject FO asymmetry A[m, n, subj] = first-half - second-half density."""
    return fo_density[:, :, 0, 0, :] - fo_density[:, :, 1, 0, :]

def best_ordering(mean_direction, n_restarts=opt_restarts, n_perms=opt_perms, seed=0):
    """The greedy optimiser is stochastic; restart it and keep the strongest
    clockwise ordering (most negative imaginary rotational momentum)."""
    mean_direction = np.nan_to_num(mean_direction)
    rng = np.random.default_rng(seed)
    best_order, best_v = None, np.inf
    for _ in range(n_restarts):
        np.random.seed(int(rng.integers(2 ** 31)))
        order = cu.optimise_sequence_stream(mean_direction, n_perms=n_perms)
        v = np.imag(np.sum(cu.circle_angles(order) * mean_direction))
        if v < best_v:
            best_v, best_order = v, order
    return best_order

def cycle_strength(asym, order):
    """Cycle strength on a fixed ordering.
    asym (K, K) -> scalar; asym (K, K, n_subj) -> per-subject vector."""
    return cu.compute_cycle_strength(cu.circle_angles(order), asym)

def _null_once(asym, seed):
    """One permutation: relabel states independently per subject (= permute the
    rows and columns of each subject's asymmetry), re-optimise the ordering and
    return the resulting group cycle strength."""
    rng = np.random.default_rng(seed)
    perm = np.empty_like(asym)
    for s in range(asym.shape[-1]):
        p = rng.permutation(K)
        perm[:, :, s] = asym[:, :, s][np.ix_(p, p)]
    md = np.nan_to_num(perm.mean(-1))
    return float(cycle_strength(md, cu.optimise_sequence_stream(md, n_perms=opt_perms_null)))

def permutation_null(asym, order, n_perm=1000, seed=0):
    """Group cycle strength on `order` vs a within-subject relabelling null.
    Returns (observed strength, p, null distribution)."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(2 ** 31, size=n_perm)
    null = np.array(Parallel(n_jobs=n_jobs)(delayed(_null_once)(asym, int(s)) for s in seeds))
    obs = float(cycle_strength(asym.mean(-1), order))
    p = (1 + np.sum(null >= obs)) / (1 + n_perm)
    return obs, p, null

def rayleigh(phases):
    """Rayleigh test of non-uniformity for a set of angles. Returns (R, p)."""
    phases = phases[~np.isnan(phases)]
    n = phases.size
    if n < 2:
        return np.nan, np.nan
    r = np.abs(np.mean(np.exp(1j * phases)))
    # Zar (1999) approximation. The Mardia series used previously has a correction
    # term that turns negative once n*r**2 is large, clipping every p-value to 0.
    res = n * r  # resultant length
    p = np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - res ** 2)) - (1 + 2 * n))
    return r, float(np.clip(p, 0, 1))

def circ_corr(a, b):
    """Circular correlation between two sets of angles."""
    a = a - np.angle(np.sum(np.exp(1j * a)))
    b = b - np.angle(np.sum(np.exp(1j * b)))
    num = np.sum(np.sin(a) * np.sin(b))
    den = np.sqrt(np.sum(np.sin(a) ** 2) * np.sum(np.sin(b) ** 2))
    return num / den if den > 0 else np.nan

# --- cache dir for the cycle analyses (shared helpers defined earlier) ---
cyc_dir = f'results/dynamics_cycles/{parcellation}'
os.makedirs(cyc_dir, exist_ok=True)


# %%
# run TINDA + group cycle ordering + edge significance (cached)
if not recompute and os.path.exists(f'{cyc_dir}/fo_asymmetry.pkl'):
    R = pickle.load(open(f'{cyc_dir}/fo_asymmetry.pkl', 'rb'))
    fo_density, fo_sum, stats = R['fo_density'], R['fo_sum'], R['stats']
    asym, mean_asym, order = R['asym'], R['mean_asym'], R['order']
    t_asym, p_asym, p_asym_bh = R['t_asym'], R['p_asym'], R['p_asym_bh']
else:
    fo_density, fo_sum, stats = cu.tinda(tcs, n_bins=2)   # fo_density: (K, K, 2, 1, n_subj)
    asym = asymmetry(fo_density)                          # (K, K, n_subj)
    mean_asym = asym.mean(-1)
    order = best_ordering(mean_asym)                      # group-level cycle ordering
    t_asym, p_asym = sps.ttest_1samp(asym, 0.0, axis=-1)  # one-sample t per directed edge
    p_asym_bh = np.ones((K, K))
    p_asym_bh[~np.eye(K, dtype=bool)] = multi.multipletests(
        p_asym[~np.eye(K, dtype=bool)], method='fdr_bh')[1]
    pickle.dump(dict(fo_density=fo_density, fo_sum=fo_sum, stats=stats,
                 asym=asym, mean_asym=mean_asym, order=order,
                 t_asym=t_asym, p_asym=p_asym, p_asym_bh=p_asym_bh), open(f'{cyc_dir}/fo_asymmetry.pkl', 'wb'))

sig = (p_asym_bh < 0.05) & (~np.eye(K, dtype=bool))
print('cycle order:', [state_names[i] for i in order])
print(f'{int(sig.sum())}/{n_edges} directed edges significant (FDR alpha=0.05)')


# %%
from palettable.cartocolors.diverging import Tropic_7_r as pmap
fomap = pmap.mpl_colormap

o = list(order)
# rows of mean_asym are the reference (interval) state, columns the state whose FO
# is measured, so transpose to put the reference on the x-axis. p_asym_bh has to be
# reordered by o alongside it, otherwise the stars land on the wrong cells.
M = mean_asym[np.ix_(o, o)].T
p_plot = p_asym_bh[np.ix_(o, o)].T
vmax = np.nanmax(np.abs(mean_asym))

fig, ax = plt.subplots(figsize=(4.3*1.25, 4.3*1.25))
im = ax.imshow(M, cmap=fomap, vmin=-vmax, vmax=vmax)
for i in range(K):
    for j in range(K):
        p = p_plot[i, j]
        star = misc.get_star(p) if p < 0.05 else ''
        ax.text(j, i+.2, star, ha='center', va='center', fontsize=15, color='black')
ax.set_xticks(range(K)); ax.set_xticklabels([misc.atoms_names[state_names[i]] for i in o], rotation=90)
ax.set_yticks(range(K)); ax.set_yticklabels([misc.atoms_names[state_names[i]] for i in o])
# make ticks colored according to the state colors
for tick in ax.get_xticklabels():
    ax.set_xticklabels([tick.get_text() for tick in ax.get_xticklabels()], color="#585858")
for tick in ax.get_yticklabels():
    key = [k for k, v in misc.atoms_names.items() if v == tick.get_text()][0]
    tick.set_color(misc.atoms_colors[key])
ax.set_xlabel('Reference mode', fontsize=rc_fontsize, color="#585858")
fig.colorbar(im, ax=ax, aspect=10, pad=0.04, shrink=0.2, anchor=(-.1, 0.875)) 
             #label='FO \n asymmetry')
for i in range(K + 1):
    ax.axvline(i - 0.5, color='black', linewidth=0.5, linestyle='-')
    ax.axhline(i - 0.5, color='black', linewidth=0.5, linestyle='--')
fig.tight_layout()

fig.savefig(f'{plots_dir}/cycle_fo_asymmetry.png', bbox_inches='tight', transparent=True)


# %%
# group cycle diagram (only edges for strong FO asymmetries)
import importlib
import cycle_utils as cu
importlib.reload(cu)

edges = np.zeros((K, K))
sig = (p_asym_bh < 0.0001) & (~np.eye(K, dtype=bool))
edges[sig & (mean_asym > 0)] = 1
edges[sig & (mean_asym < 0)] = -1

fig, ax = cu.plot_cycle(list(order), mean_asym, edges, halo_scale=1.1, band_alpha=0.9,
                        color_scheme=node_colors,
                        state_names={number: '' for number in range(K)}, new_figure=True)
fig.savefig(f'{plots_dir}/cycle_diagram_group.png', bbox_inches='tight', transparent=True)


# %%
# observed cycle strength (per subject + group) vs the permutation null (cached)
# +1 = perfect clockwise, -1 = perfect anticlockwise, 0 = stochastic/no cycle. The
# null relabels states independently per subject, re-optimises the ordering and
# recomputes the group strength.
if not recompute and os.path.exists(f'{cyc_dir}/cycle_strength.pkl'):
    R = pickle.load(open(f'{cyc_dir}/cycle_strength.pkl', 'rb'))
    strength_subj, S_obs, S_p, S_null = R['strength_subj'], R['S_obs'], R['S_p'], R['S_null']
else:
    strength_subj = cycle_strength(asym, order)
    S_obs, S_p, S_null = permutation_null(asym, order, n_perm=1000)
    pickle.dump(dict(strength_subj=strength_subj, S_obs=S_obs, S_p=S_p, S_null=S_null), open(f'{cyc_dir}/cycle_strength.pkl', 'wb'))
print(f'group S={S_obs:.4f}  null={S_null.mean():.4f}+-{S_null.std():.4f}  p={S_p:.4g}')
print(f'per-subject S={strength_subj.mean():.4f}+-{strength_subj.std():.4f}, '
      f'{(strength_subj > 0).sum()}/{len(strength_subj)} positive')
print(f'weakest subject S={strength_subj.min():.4f} vs strongest null perm S={S_null.max():.4f}')


# %%
from scipy.stats import gaussian_kde
fig, ax = plt.subplots(1, 1, figsize=(1.7*1.25, .8*1.25))
lo = min(S_null.min(), strength_subj.min())
hi = max(S_null.max(), strength_subj.max())
pad = 0.1 * (hi - lo)                       # extend so the tails reach ~0
grid = np.linspace(lo - pad, hi + pad, 400)
for vals, color, label in [(S_null, '0.6', 'Null'),
                           (strength_subj, '#f9919a', 'Subjects')]:
    dens = gaussian_kde(vals)(grid)
    ax.plot(grid, dens / dens.max(), color=color, label=label)   # peak-normalised to 1
ax.axvline(S_obs, color='crimson', lw=1.5, label='Group')
ax.set_ylim(0, 1.05)
ax.set_xlim(left=0)
ax.set_xlabel('Cycle Strength')
#ax.set_ylabel('Density')
sns.despine(ax=ax)
ax.xaxis.set_major_locator(plt.MaxNLocator(2))
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=(1, 0), frameon=False,
          handlelength=1.0, handletextpad=0.4, labelspacing=0.1)
ax.set_xticks([0, 0.01, 0.02]); ax.set_xticklabels(['0', '0.01', '0.02'])
ax.set_yticks([0, 1])
fig.savefig(f'{plots_dir}/cycle_strength.png', bbox_inches='tight', transparent=True)


# %%
# per-subject cycle duration (median pooled revisit interval) and rate (same ting rlly)
if not recompute and os.path.exists(f'{cyc_dir}/duration_rate.pkl'):
    R = pickle.load(open(f'{cyc_dir}/duration_rate.pkl', 'rb'))
    duration, rate, dur_state = R['duration'], R['rate'], R['dur_state']
else:
    duration = np.full(len(stats), np.nan)
    dur_state = np.full((len(stats), K), np.nan)
    for si, st in enumerate(stats):
        pooled = []
        for k in range(K):
            d = st[k]['durations']
            if len(d):
                dur_state[si, k] = np.median(d) * tr
                pooled.append(np.asarray(d) * tr)
        if pooled:
            duration[si] = np.median(np.concatenate(pooled))
    rate = 1.0 / duration
    pickle.dump(dict(duration=duration, rate=rate, dur_state=dur_state), open(f'{cyc_dir}/duration_rate.pkl', 'wb'))
print(f'cycle duration: median {np.nanmedian(duration):.1f}s '
      f'(mean {np.nanmean(duration):.1f}+-{np.nanstd(duration):.1f}s)')
print(f'cycle rate: mean {np.nanmean(rate):.4f}+-{np.nanstd(rate):.4f} Hz')


# %%
plot_size = (3.5, 1.75)
fig1, ax1 = plt.subplots(figsize=plot_size)
ax1.hist(duration, bins=20, color='crimson', alpha=0.8, cumulative=True)
ax1.set_xlabel('Cycle duration (s)')
ax1.set_ylabel('Subjects')
sns.despine(ax=ax1)
fig1.tight_layout()
fig1.savefig(f'{plots_dir}/cycle_duration.png', bbox_inches='tight')
plt.close(fig1)
fig2, ax2 = plt.subplots(figsize=plot_size)
ax2.hist(rate, bins=20, color='crimson', alpha=0.8, cumulative=True)
ax2.set_xlabel('Cycle rate (Hz)')
ax2.set_ylabel('Subjects')
sns.despine(ax=ax2)
fig2.tight_layout()
fig2.savefig(f'{plots_dir}/cycle_rate.png', bbox_inches='tight')


# %%
# per-subject preferred phase of each state -> circular concentration across subjects (cached)
if not recompute and os.path.exists(f'{cyc_dir}/phase.pkl'):
    R = pickle.load(open(f'{cyc_dir}/phase.pkl', 'rb'))
    theta, phase = R['theta'], R['phase']
    phase_R, phase_p, ordering_cc = R['phase_R'], R['phase_p'], R['ordering_cc']
else:
    theta = np.zeros(K)
    for pos, st in enumerate(order):
        theta[st] = 2 * np.pi * pos / K

    phase = np.full((K, len(tcs)), np.nan)
    for s in range(len(tcs)):
        for n in range(K):
            w = np.nan_to_num(asym[:, n, s]); w[n] = 0.0
            vec = np.sum(w * np.exp(1j * theta))
            if np.abs(vec) > 0:
                phase[n, s] = np.angle(vec)

    phase_R = np.full(K, np.nan); phase_p = np.full(K, np.nan)
    for n in range(K):
        phase_R[n], phase_p[n] = rayleigh(phase[n])

    # per-subject ordering vs group ordering (circular correlation of state positions)
    def _agree(s):
        os_ = best_ordering(asym[:, :, s], n_restarts=8, n_perms=2 * 10 ** 4, seed=s)
        th = np.zeros(K)
        for pos, st in enumerate(os_):
            th[st] = 2 * np.pi * pos / K
        return circ_corr(theta, th)
    ordering_cc = np.array(Parallel(n_jobs=n_jobs)(delayed(_agree)(s) for s in range(len(tcs))))
    pickle.dump(dict(theta=theta, phase=phase, phase_R=phase_R, phase_p=phase_p,
                 ordering_cc=ordering_cc), open(f'{cyc_dir}/phase.pkl', 'wb'))

print(f'{int((phase_p < 0.05).sum())}/{K} states Rayleigh-significant; mean R={np.nanmean(phase_R):.3f}')
print(f'weakest state: R={phase_R[np.nanargmin(phase_R)]:.3f}, '
      f'p={misc.write_scientific(np.nanmax(phase_p))} (all ten modes are below this)')
print(f'per-subject ordering circ-corr to group: mean={np.nanmean(ordering_cc):.3f}')
print(phase_p)


# %%
# State-wise phase consistency
fig, ax = plt.subplots(figsize=(4, 2.5))
o = np.argsort(phase_R)[::-1]

# Extract the colors for each state in the sorted order
colors = [misc.atoms_colors[state_names[i]] for i in o]

# Create the lollipop plot: vlines for the stems, scatter for the markers
#ax.vlines(x=range(K), ymin=0, ymax=phase_R[o], colors='lightgrey', linewidth=2)
#ax.scatter(range(K), phase_R[o], color=colors, s=50, zorder=3, edgecolors='black', linewidth=0.5)
ax.bar(range(K), phase_R[o], color=colors, edgecolor='black', linewidth=0.5, zorder=3)
for rank, i in enumerate(o):
    ax.text(rank, phase_R[i] + 0.05, misc.get_star(phase_p[i]), ha='center', fontsize=13)
ax.set_xticks(range(K)); ax.set_xticklabels([misc.atoms_names[state_names[i]] for i in o], rotation=90)
ax.set_ylabel('Phase \n Consistency')
ax.set_ylim(0,1.1)
fig.tight_layout()
sns.despine()

fig.savefig(f'{plots_dir}/cycle_phase_consistency.png', bbox_inches='tight')


# %%
# does the asymmetry structure cluster by film?
# distance-based, non-parametric PERMANOVA: a subject-by-subject distance matrix
# (1 - Pearson r between asymmetry vectors) is partitioned into within- vs between-
# film dispersion, pseudo-F = between/within, significance by permuting film labels
# (only assumption: exchangeability under the null). Robust to high dimensionality
# and the unbalanced film sizes (n=20, 18 vs eight n=6).
if not recompute and os.path.exists(f'{cyc_dir}/permanova.pkl'):
    R = pickle.load(open(f'{cyc_dir}/permanova.pkl', 'rb'))
    C, F_obs, F_p, within_r, between_r = R['C'], R['F_obs'], R['F_p'], R['within_r'], R['between_r']
else:
    off = ~np.eye(K, dtype=bool)
    V = np.array([asym[:, :, s][off] for s in range(len(tcs))])
    C = np.corrcoef(V)
    D2 = (1.0 - C) ** 2
    labels = np.array(films)
    groups = np.unique(labels)

    def pseudo_f(lab):
        ss_total = D2.sum() / (2 * len(lab))
        ss_within = sum(D2[np.ix_(np.where(lab == g)[0], np.where(lab == g)[0])].sum()
                        / (2 * (lab == g).sum()) for g in groups if (lab == g).sum() > 1)
        return ((ss_total - ss_within) / (len(groups) - 1)) / (ss_within / (len(lab) - len(groups)))

    rng = np.random.default_rng(0)
    F_obs = pseudo_f(labels)
    F_null = np.array([pseudo_f(rng.permutation(labels)) for _ in range(10000)])
    F_p = (1 + np.sum(F_null >= F_obs)) / (1 + len(F_null))
    within_r = np.mean([C[i, j] for i in range(len(tcs)) for j in range(i + 1, len(tcs)) if labels[i] == labels[j]])
    between_r = np.mean([C[i, j] for i in range(len(tcs)) for j in range(i + 1, len(tcs)) if labels[i] != labels[j]])
    pickle.dump(dict(C=C, F_obs=F_obs, F_p=F_p, within_r=within_r, between_r=between_r), open(f'{cyc_dir}/permanova.pkl', 'wb'))

print(f'PERMANOVA pseudo-F={F_obs:.3f}  p={F_p:.4f}')
print(f'within-film r={within_r:.3f}  vs  between-film r={between_r:.3f}')

iu_all = np.triu_indices(C.shape[0], 1)
print(f'asym. similarity over all {len(iu_all[0])} subject pairs: '
      f'r={C[iu_all].mean():.3f}+-{C[iu_all].std():.3f}, '
      f'{(C[iu_all] > 0).sum()}/{len(iu_all[0])} positive')

# a non-significant PERMANOVA is only absence of evidence, so back it with a Bayes
# factor for no film effect, and with the cycle strength of each film on its own
from scipy.stats import f_oneway, ttest_1samp

n_sub = len(tcs)
C_nan = C.copy()
np.fill_diagonal(C_nan, np.nan)
r_within = np.array([np.nanmean(C_nan[i, films == films[i]]) for i in range(n_sub)])
r_between = np.array([np.nanmean(C_nan[i, films != films[i]]) for i in range(n_sub)])
t, p, cil, ciu, d, bf, dof, power = misc.aug_t_test(r_within, r_between, paired=True)

print(f'\nasym. similarity, within-film r={r_within.mean():.3f}+-{r_within.std():.3f} vs '
      f'between-film r={r_between.mean():.3f}+-{r_between.std():.3f}')
print(f'paired T={misc.write_scientific(t)}, {misc.write_p(p)}, '
      f'd={misc.write_scientific(d)}')
print(f'BF01={misc.write_scientific(1 / bf)} (evidence for the absence of a film effect)')

print('\ncycle strength per film:')
for film in np.unique(films):
    s = strength_subj[films == film]
    tt, pp = ttest_1samp(s, 0.0)
    print(f'  {film}: n={len(s)}, S={s.mean():.4f}+-{s.std():.4f}, '
          f'{misc.write_p(pp)}, {(s > 0).sum()}/{len(s)} positive')

film_groups = [strength_subj[films == film] for film in np.unique(films)]
film_means = [s.mean() for s in film_groups]
F_s, p_s = f_oneway(*film_groups)
print(f'\nwithin-film S ranges {min(film_means):.4f} to {max(film_means):.4f}')
print(f'strongest null permutation was S={S_null.max():.4f}')
print(f'strength does not differ across films: '
      f'F({len(film_groups) - 1},{n_sub - len(film_groups)})='
      f'{misc.write_scientific(F_s)}, {misc.write_p(p_s)}')


import matplotlib.patches as patches

# subject-by-subject asymmetry correlation sorted by film
si = np.argsort(films)
fl = films[si]
unique_vals, block_sizes = np.unique(fl, return_counts=True)
block_edges = np.insert(np.cumsum(block_sizes), 0, 0)
fig, ax = plt.subplots(figsize=(7*.7, 6*.7))
im = ax.imshow(C[np.ix_(si, si)], cmap='bwr', vmin=-1, vmax=1)
for i in range(len(block_edges) - 1):
    rect = patches.Rectangle(
        (block_edges[i] - 0.5, block_edges[i] - 0.5), block_sizes[i], block_sizes[i],
        linewidth=.8, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
ax.set_xlabel('Subject'); ax.set_ylabel('Subject')
ax.xaxis.set_major_locator(plt.MaxNLocator(2))
ax.yaxis.set_major_locator(plt.MaxNLocator(2))
fig.colorbar(im, ax=ax, label='$r$', aspect=10, pad=0.04, shrink=0.2, anchor=(-.1, 0.95))
fig.tight_layout()
fig.savefig(f'{plots_dir}/cycle_film_similarity_matrix.png', bbox_inches='tight')

# within vs between film
within = [C[i, j] for i in range(len(tcs)) for j in range(i + 1, len(tcs)) if films[i] == films[j]]
between = [C[i, j] for i in range(len(tcs)) for j in range(i + 1, len(tcs)) if films[i] != films[j]]

fig, ax = plt.subplots(figsize=(1.75, 3.2))
sns.violinplot([within, between], ax=ax, palette=['lightgrey', 'lightgrey'], inner=None, cut=0, split=True)
ax.set_xticklabels(['Shared stimuli', 'Different'], rotation=90)
ax.set_ylabel(r'FO asymm. sim.')
ax.set_ylim(0, 1)
fig.tight_layout()
sns.despine(ax=ax)
fig.savefig(f'{plots_dir}/cycle_film_similarity_violin.png', bbox_inches='tight')


# %%
# cycle phase synchrony: same-film vs different-film (cached)
import matplotlib.patches as patches

if not recompute and os.path.exists(f'{cyc_dir}/synchrony.pkl'):
    R = pickle.load(open(f'{cyc_dir}/synchrony.pkl', 'rb'))
    PLV, PLV_null, T = R['PLV'], R['PLV_null'], R['T']
    within_plv, between_plv, within_null_plv = R['within_plv'], R['between_plv'], R['within_null_plv']
    sync_obs, sync_p, sync_null = R['sync_obs'], R['sync_p'], R['sync_null']
    obs_w, shift_p, shift_null = R['obs_w'], R['shift_p'], R['shift_null']
else:
    theta = np.array([2 * np.pi * np.where(order == k)[0][0] / K for k in range(K)])  # state -> phase
    T = min(map(len, tcs))
    seqs = np.array([np.asarray(_w['sequences'][int(s)])[:T].astype(int) - 1 for s in subject_ids])
    z = np.exp(1j * theta[seqs])                 # (N, T) phase as unit vectors

    # pairwise phase-locking value (time-averaged) = pairwise Kuramoto coherence
    PLV = np.abs(z @ z.conj().T) / T             # (N, N); |<exp(i(phi_i - phi_j))>_t|
    np.fill_diagonal(PLV, np.nan)

    iu = np.triu_indices(len(subject_ids), 1)
    same = (films[:, None] == films[None, :])[iu]
    within_plv, between_plv = PLV[iu][same], PLV[iu][~same]

    # is within-film synchrony > between-film? permute film labels.
    def plv_gap(lab):
        sm = (lab[:, None] == lab[None, :])[iu]
        return np.nanmean(PLV[iu][sm]) - np.nanmean(PLV[iu][~sm])
    rng = np.random.default_rng(0)
    sync_obs = plv_gap(films)
    sync_null = np.array([plv_gap(rng.permutation(films)) for _ in range(10000)])
    sync_p = (1 + np.sum(sync_null >= sync_obs)) / (1 + len(sync_null))

    # within-film synchrony vs circular-shift null (preserves occupancy + autocorr,
    # destroys cross-subject time-locking). One loop yields the per-realisation
    # within-film mean (p-value) and the per-pair null PLV (for the violin).
    def within_mean(P):
        return np.nanmean([P[np.ix_(idx, idx)][np.triu_indices(idx.size, 1)].mean()
                           for f in np.unique(films)
                           for idx in [np.where(films == f)[0]] if idx.size > 1])
    obs_w = within_mean(PLV)
    n_shift = 1000
    acc = np.zeros_like(PLV)
    shift_null = np.empty(n_shift)
    for i in range(n_shift):
        zs = np.array([np.roll(z[s], rng.integers(T)) for s in range(len(subject_ids))])
        P = np.abs(zs @ zs.conj().T) / T
        np.fill_diagonal(P, np.nan)
        acc += np.nan_to_num(P)
        shift_null[i] = within_mean(P)
    PLV_null = acc / n_shift
    np.fill_diagonal(PLV_null, np.nan)
    within_null_plv = PLV_null[iu][same]
    shift_p = (1 + np.sum(shift_null >= obs_w)) / (1 + n_shift)
    pickle.dump(dict(PLV=PLV, PLV_null=PLV_null, T=T,
                 within_plv=within_plv, between_plv=between_plv, within_null_plv=within_null_plv,
                 sync_obs=sync_obs, sync_p=sync_p, sync_null=sync_null,
                 obs_w=obs_w, shift_p=shift_p, shift_null=shift_null), open(f'{cyc_dir}/synchrony.pkl', 'wb'))

# subject-level contrast: each participant gives one within-film and one between-film
# mean PLV. testing over the 3655 pairs would be pseudoreplication, they share subjects
n_sub = len(subject_ids)
plv_within = np.array([np.nanmean(PLV[i, films == films[i]]) for i in range(n_sub)])
plv_between = np.array([np.nanmean(PLV[i, films != films[i]]) for i in range(n_sub)])
plv_null = np.array([np.nanmean(PLV_null[i, films == films[i]]) for i in range(n_sub)])

print(f'T={T} samples')
print(f'same film {plv_within.mean():.3f}+-{plv_within.std():.3f}, '
      f'different film {plv_between.mean():.3f}+-{plv_between.std():.3f}, '
      f'circ-shift null {plv_null.mean():.3f}+-{plv_null.std():.3f}')

for label, a, b, p_perm in [('same vs different film', plv_within, plv_between, sync_p),
                            ('same film vs circ-shift null', plv_within, plv_null, shift_p)]:
    t, p, cil, ciu, d, bf, dof, power = misc.aug_t_test(a, b, paired=True)
    print(f'\n{label}')
    print(f'  paired T({int(dof)})={misc.write_scientific(t)}, {misc.write_p(p)}, '
          f'd={misc.write_scientific(d)}, BF10={misc.write_scientific(bf)}')
    print(f'  95% CI [{misc.write_scientific(cil)}, {misc.write_scientific(ciu)}], '
          f'{(a > b).sum()}/{n_sub} participants')
    print(f'  {misc.write_p(p_perm, floor=1e-4, sub="perm")}')


# %%
# cycle-phase synchrony: matrix (sorted by film) + same / different / null
si = np.argsort(films); fl = films[si]

from matplotlib.colors import LinearSegmentedColormap
indexes = [0, 0.5, 0.75, 1]
colors = ['white', 'white', '#008080', '#004c4c']
state_cmap = LinearSegmentedColormap.from_list('custom_plv', list(zip(indexes, colors)))

_, bs = np.unique(fl, return_counts=True); ed = np.insert(np.cumsum(bs), 0, 0)
plv_pl = PLV[np.ix_(si, si)]
plv_pl[np.eye(len(plv_pl), dtype=bool)] = np.nanmax(plv_pl)
fig, ax = plt.subplots(1, 1, figsize=(2.8*1.25, 2.8*1.25))
im = ax.imshow(plv_pl, cmap=state_cmap, vmin=0, vmax=.1)
for i in range(len(bs)):
    ax.add_patch(patches.Rectangle((ed[i] - 0.5, ed[i] - 0.5), bs[i], bs[i],
                      lw=.8, edgecolor='black', facecolor='none'))
ax.set_xlabel('Subject'); ax.set_ylabel('Subject')
ax.yaxis.set_label_coords(-0.05, 0.72)
ax.xaxis.set_label_coords(0.6, -0.15)
ax.set_xticks([0, 50]); ax.set_yticks([0, 50])
fig.colorbar(im, ax=ax, #label='Cycle \n Synchrony',
             aspect=8, pad=0.04, shrink=0.2, anchor=(-.1, 0.87))
fig.savefig(f'{plots_dir}/cycle_synchrony_matrix.png', bbox_inches='tight')


# %%
# fig, ax = plt.subplots(1, 1, figsize=(2.25, 4))
# sns.boxplot([within_plv, between_plv, within_null_plv], ax=ax,
#             showfliers=False, width=0.6,
#             showcaps=False, boxprops=dict(facecolor='none', edgecolor='black', linewidth=1))
# sns.stripplot([within_plv, between_plv, within_null_plv], ax=ax,
#               palette=['#004c4c', '#008080', 'grey'], alpha=0.05, size=3, jitter=0.1, zorder=-1)
# ax.set_xticks([0, 1, 2])
# ax.set_xticklabels(['Shared stimuli', 'Different', 'Null'], rotation=90)
# ax.get_xticklabels()[-1].set_color('grey')
# ax.set_ylabel('Cycle Synchrony')
# sns.despine(ax=ax); fig.tight_layout()
# ax.yaxis.set_major_locator(plt.MaxNLocator(2))
# fig.savefig(f'{plots_dir}/cycle_synchrony_box.png', bbox_inches='tight')

# alternative plotting

# horizontal synchrony boxplot: shared-stimuli vs different film (null omitted)
fig, ax = plt.subplots(1, 1, figsize=(2.25*1.25, 1.45))
data = [within_plv, between_plv]
colors = ["#48b2b2ff", "#D0F8F8"]  # dark = shared stimuli, light = different
sns.boxplot(data=data, ax=ax, orient='h', width=0.6, showfliers=False, showcaps=False,
            palette=colors, boxprops=dict(edgecolor='black', linewidth=1),
            medianprops=dict(color='black', linewidth=1),
            whiskerprops=dict(color='black', linewidth=1))
ax.set_xticks([0, 0.06])
ax.set_xticklabels(['0', '0.06'])
ax.set_yticks([])
#ax.set_xlabel('Cycle Synchrony')
ax.xaxis.set_major_locator(plt.MaxNLocator(2))
sns.despine(ax=ax, left=True, bottom=True)

## significance bracket: use the film-label permutation p (sync_p), not a naive pairwise test
#xr = max(np.nanmax(within_plv), np.nanmax(between_plv))
#xb = xr * .8
#ax.text(xb + xr * 0.02, 0.5, misc.get_star(sync_p), va='center', ha='left', fontsize=15, rotation=90)
#ax.set_xlim(right=xb + xr * 0.13)
fig.tight_layout()
fig.savefig(f'{plots_dir}/cycle_synchrony_box.png', bbox_inches='tight', transparent=True)


# %%
tpm_off = pickle.load(open(f'{winner_dir}/mean_tpm.pkl', 'rb'))['mean_tpm']
tpm_self = pickle.load(open(f'{winner_dir}/mean_tpm_self.pkl', 'rb'))['mean_tpm_self']
cyc = pickle.load(open(f'{cyc_dir}/fo_asymmetry.pkl', 'rb'))
cycle_order = cyc['order']

# temporal itinerary: reverse the circle ordering, then rotate to start at Decryption
# so the printed sequence matches the one quoted in the Results
itinerary = list(cycle_order[::-1])
start = itinerary.index(modes.index('Decryption'))
itinerary = itinerary[start:] + itinerary[:start]

modes_cyc = [modes[i] for i in itinerary]
print('itinerary: ' + ' -> '.join(misc.atoms_texcommands[m] for m in modes_cyc))

cyc_labels = [misc.atoms_names[m] for m in modes_cyc]
reorder = np.ix_(itinerary, itinerary)

panels = [
    (tpm_off[reorder], 'Excluding self-transitions'),
    (tpm_self[reorder], 'Including self-transitions'),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, (mat, title) in zip(axes, panels):
    im = ax.imshow(mat, cmap=cmr.horizon)
    ax.set_title(title, fontsize=rc_fontsize - 1, pad=10)
    ax.set_xticks(np.arange(len(modes_cyc)))
    ax.set_xticklabels(cyc_labels, rotation=90)
    ax.set_yticks(np.arange(len(modes_cyc)))
    ax.set_yticklabels(cyc_labels)
    ax.set_xlabel('$X_{t+1}$')
    ax.set_ylabel('$X_t$')
    for tl, m in zip(ax.get_xticklabels(), modes_cyc):
        tl.set_color(misc.atoms_colors[m])
    for tl, m in zip(ax.get_yticklabels(), modes_cyc):
        tl.set_color(misc.atoms_colors[m])
    for i in range(len(modes_cyc) + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5)
        ax.axvline(i - 0.5, color='black', linewidth=0.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    cbar.ax.set_title('$P(X_{t+1}|X_t)$', fontsize=rc_fontsize - 4, pad=8)

plt.tight_layout()
#fig.savefig(f'{plots_dir}/supp_tpm.png', bbox_inches='tight')
plt.show()
