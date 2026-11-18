# %%
# libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import warnings
import cmasher as cmr
import palettable as pl
import pingouin as pg
from scipy.stats import spearmanr
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

root_dir = os.path.abspath('..')
if root_dir not in sys.path:
    sys.path.append(root_dir)  # network_utils, wang_utils and friends live at the repo root
# misc still resolves to replication/misc.py: the notebook's own directory comes first
plt.rcParams.update({'font.size': 15})

import network_utils as nu
import misc

parcellation = misc.parcellation
n_regions = misc.n_regions

MRI_path = '/home/george/Research/fMRI_general'
atlas_info = pd.read_csv(f'{MRI_path}/atlases/schaefer_extend_atlases/{parcellation}.csv')

main_dir = f'nndb2_results/main'

plots_dir = f'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

networks_dir = f'nndb2_results/networks/{parcellation}'
if not os.path.exists(networks_dir):
    os.makedirs(networks_dir)

import sys
sys.path.append(f'{MRI_path}')
from cortexcanvas import (plot_wholebrain, plot_full, 
                          plot_wholebrain_row, plot_wholebrain_row_2,
                          plot_grand_ensemble, plot_brain_network, plot_flatex)

from palettable.scientific.diverging import Roma_5_r, Vik_3, Berlin_20

def load_connectome(perp, atom_indices, fix_nans=True):
    mat = np.load(f'{main_dir}/{parcellation}/sub-{perp}/phiid_matrix.npy')
    out = mat[:, :, atom_indices].sum(axis=2).astype(np.float64)
    if fix_nans:
        if np.isnan(out).any():
            min_finite = np.nanmin(out)
            out = np.where(np.isnan(out), min_finite, out)
    return out


indidiff_results_dir = f'nndb2_results/indidiffs/{parcellation}'
os.makedirs(indidiff_results_dir, exist_ok=True)


# %%
# PCA over participant x atom edge profiles
from sklearn.decomposition import PCA
from matplotlib.colors import LinearSegmentedColormap

n_atoms = len(misc.atom_order)
cache_path = f'{networks_dir}/pca_subject_atom.npy'

if not os.path.exists(cache_path):
    triu_idx = np.triu_indices(n_regions, k=1)
    rows = []
    for sub_id in tqdm(misc.subjects, desc='subjects'):
        for atom_name in misc.atom_order:
            mat = load_connectome(sub_id, misc.atoms[atom_name])
            rows.append(mat[triu_idx])
    X = np.vstack(rows)
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X)
    evr = pca.explained_variance_ratio_
    np.save(cache_path, {'Xp': Xp, 'evr': evr})
else:
    cached = np.load(cache_path, allow_pickle=True).item()
    Xp = cached['Xp']; evr = cached['evr']

fig, ax = plt.subplots(1, 1, figsize=(3.25, 3.25))
for i, atom in enumerate(misc.atom_order):
    atom_vals = Xp[i::n_atoms]
    color = misc.atoms_colors[atom]
    ccmap = LinearSegmentedColormap.from_list('custom_cmap', ['white', color])
    sns.kdeplot(x=atom_vals[:, 1], y=atom_vals[:, 0], ax=ax,
                levels=1000, cmap=ccmap, fill=True, zorder=-1,
                thresh=0, alpha=1)
    ax.scatter(atom_vals[:, 1], atom_vals[:, 0], s=10, color=color,
               alpha=1, edgecolor="k", linewidth=0.25, zorder=1)

# make lower y limit bigger
ax.set_ylim(ax.get_ylim()[0] * 1.2, ax.get_ylim()[1])
#ax.set_ylabel(f'Latent D1 ({evr[0] * 100:.1f}%)')
#ax.set_xlabel(f'Latent D2 ({evr[1] * 100:.1f}%)')
ax.set_ylabel(f'PC1', labelpad=-20)
ax.set_xlabel(f'PC2')
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
sns.despine()
# print variance explained
print(f'Variance explained: PC1: {evr[0] * 100:.1f}%, PC2: {evr[1] * 100:.1f}%')

# save figure
fig.savefig(f'{plots_dir}/pca_subject_atom.png', dpi=800, bbox_inches='tight')

plt.tight_layout()
plt.show()


# %%

# s.d across individuals for each region, averaged across edges

for atom_name in tqdm(misc.atom_order, desc='SD topo'):
    cache_path = f'{indidiff_results_dir}/nodal_sd_{atom_name}.npy'

    if os.path.exists(cache_path):
        nodal_sd = np.load(cache_path)
    else:
        stack = np.stack([
            load_connectome(perp, misc.atoms[atom_name])
            for perp in misc.subjects
        ], axis=0)  # (n_subjects, n_regions, n_regions)
        std_mat = np.std(stack, axis=0)
        nodal_sd = np.nanmean(std_mat, axis=0)
        np.save(cache_path, nodal_sd.astype(np.float32))

    vmin = np.nanpercentile(nodal_sd, 5)
    vmax = np.nanpercentile(nodal_sd, 95)
    fig = plot_wholebrain_row(nodal_sd, atlas=parcellation, brightness=0.4,
                              cmap=misc.con_cmap, vmin=vmin, vmax=vmax)
    # fig.savefig(f'{plots_dir}/sd_nodal_{atom_name}.png', bbox_inches='tight')
    #plt.close(fig)

print('SD topographic plots done.')


# %%
from matplotlib.gridspec import GridSpec
import tempfile
import hcp_utils as hcp
from PIL import Image
from cortexcanvas._plotting import _render_flatmap, atlas_loader as _atlas_loader, _crop_to_content

def _top_edges_icc(W, n_keep=1000):
    W = np.where(np.isnan(W), np.nanmin(W), W)
    ranked = nu.rank_normalise(W)
    triu = np.triu_indices(W.shape[0], k=1)
    top_idx = np.argsort(ranked[triu])[::-1][:n_keep]
    out = np.zeros_like(W, dtype=float)
    r, c = triu[0][top_idx], triu[1][top_idx]
    out[r, c] = ranked[triu[0][top_idx], triu[1][top_idx]]
    out[c, r] = out[r, c]
    return out

coords_arr = atlas_info[['x', 'y', 'z']].values[:n_regions]

all_nodal_icc = []
for atom_name in misc.atoms_nr:
    icc = np.load(f'{indidiff_results_dir}/icc_{atom_name}.npy')
    all_nodal_icc.append(np.nanmean(icc, axis=0))

plotcbar = True

pooled = np.concatenate(all_nodal_icc)
vmin = np.nanpercentile(pooled, 5)
vmax = np.nanpercentile(pooled, 95)

import colorcet as cc
cmap_ = cc.cm.diverging_rainbow_bgymr_45_85_c67

def plot_icc_composite(atom_name, matrix, nodal_icc, coords,
                       flat_vmin, flat_vmax,
                       atlas=parcellation,
                       figsize=(12, 8)):
    cmap_flat = cc.cm.diverging_rainbow_bgymr_45_85_c67

    data = nodal_icc.astype(float)
    vmin, vmax = flat_vmin, flat_vmax
    if np.nanmean(data) < 1:
        data, vmin, vmax = data * 100, vmin * 100, vmax * 100

    atlas_L_raw, atlas_R_raw, _, _ = _atlas_loader(atlas)

    x_r = coords[:, 0].max() - coords[:, 0].min()
    y_r = coords[:, 1].max() - coords[:, 1].min()
    net_w = (x_r / y_r) * figsize[1]

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 2, figure=fig, width_ratios=[net_w*.95, figsize[0] - net_w], wspace=-0.1)
    ax_net = fig.add_subplot(gs[0])
    gs_r = gs[1].subgridspec(2, 1, hspace=0.02)
    ax_rh = fig.add_subplot(gs_r[1])
    ax_lh = fig.add_subplot(gs_r[0])

    with tempfile.TemporaryDirectory() as tmp:
        fig_net, _ = plot_brain_network(
            matrix, coords=coords,
            edge_scores=matrix,
            node_scores=np.nansum(matrix, axis=0),
            view='top',
            figsize=(8, 8),
            node_color=misc.atoms_colors[atom_name],
            edge_color='#2A2A2A',
            node_edgecolor='#3A3939',
            node_linewidth=0.5,
            edge_alpha=(0, 1),
            edge_width=(0.1, 3.0),
            node_size=(25, 300),
        )
        net_path = os.path.join(tmp, 'net.png')
        fig_net.savefig(net_path, dpi=600, bbox_inches='tight', transparent=True)
        plt.close(fig_net)
        net_img = _crop_to_content(Image.open(net_path))
        ax_net.imshow(net_img, aspect='auto')
        ax_net.axis('off')
        ax_net.set_zorder(2)

        for mesh, atlas_raw, sulc, fname, ax in [
            (hcp.mesh.flat_left,  atlas_L_raw, hcp.mesh.sulc_left,  'lh.png', ax_lh),
            (hcp.mesh.flat_right, atlas_R_raw, hcp.mesh.sulc_right, 'rh.png', ax_rh),
        ]:
            fig_f = _render_flatmap(mesh, atlas_raw, data, cmap_flat, vmin, vmax, sulc=sulc)
            path = os.path.join(tmp, fname)
            fig_f.savefig(path, dpi=800, bbox_inches='tight')
            plt.close(fig_f)
            ax.imshow(Image.open(path))
            ax.axis('off')

    return fig

for atom_name in misc.atoms_nr:
    icc = np.load(f'{indidiff_results_dir}/icc_{atom_name}.npy')
    matrix = _top_edges_icc(icc)
    nodal_icc = np.nanmean(icc, axis=0)

    fig = plot_icc_composite(atom_name, matrix, nodal_icc, coords_arr,
                             flat_vmin=vmin, flat_vmax=vmax)
    fig.savefig(f'{plots_dir}/icc_composite_{atom_name}.png', dpi=800, bbox_inches='tight')
    plt.close(fig)

dp = 2 if f'{vmin:.2f}' != f'{vmax:.2f}' else 3
fig, ax = plt.subplots(1, 1, figsize=(.1, .6))
cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap_), cax=ax, orientation='vertical')
cbar.set_ticks([0, 1])
cbar.set_ticklabels([f'{vmin:.{dp}f}', f'{vmax:.{dp}f}'], fontsize=13)
fig.savefig(f'{plots_dir}/icc_topo_cbar.png', dpi=800,
            bbox_inches='tight', transparent=True)
plt.close(fig)


# %%
# write min and max ICC values
print('-' * 100)
print(f'Min and max ICC values across all atoms:')
print('-' * 100)
min_icc, max_icc = np.inf, -np.inf
for atom_name in misc.atom_order:
    icc = np.load(f'{indidiff_results_dir}/icc_{atom_name}.npy').flatten()
    print(f'{atom_name}: min = {np.nanmin(icc):.4f}, max = {np.nanmax(icc):.4f}')
    min_icc = np.minimum(min_icc, np.nanmin(icc))
    max_icc = np.maximum(max_icc, np.nanmax(icc))
print(f'Overall: min = {min_icc:.4f}, max = {max_icc:.4f}')

# flatten values of ICC and then do an ANOVA between atoms
icc_df = []
for atom_name in misc.atom_order:
    icc = np.load(f'{indidiff_results_dir}/icc_{atom_name}.npy').flatten()
    icc_df.append(pd.DataFrame({'atom': atom_name, 'icc': icc}))
icc_df = pd.concat(icc_df, axis=0)
anova_res = pg.anova(data=icc_df, dv='icc', between='atom', detailed=True)
F = anova_res['F'][0]
p = anova_res['p_unc'][0]
print('-' * 100)
F_str = misc.write_scientific(F)
p_str = misc.write_p(p)
print(f'ANOVA: $F=${F_str}, {p_str}')

# mean ICC order and pairwise comparisons between atoms
from itertools import combinations

mean_icc = icc_df.groupby('atom')['icc'].mean().reindex(misc.atom_order)
order = mean_icc.sort_values(ascending=False)
print('-' * 100)
print('Atom order by mean ICC:')
print(' > '.join(f'{misc.atoms_texcommands[a]}' for a, v in order.items()))

pair_rows = []
for atom_a, atom_b in combinations(misc.atom_order, 2):
    vals_a = icc_df[icc_df['atom'] == atom_a]['icc'].dropna().values
    vals_b = icc_df[icc_df['atom'] == atom_b]['icc'].dropna().values
    t, p_pair, cil, ciu, d, bf, dof, power = misc.aug_t_test(vals_a, vals_b, paired=False)
    pair_rows.append({
        'atom_a': atom_a, 'atom_b': atom_b,
        't': t, 'p': p_pair, 'cil': cil, 'ciu': ciu,
        'd': d, 'bf': bf, 'dof': dof, 'power': power,
    })
icc_stats_df = pd.DataFrame(pair_rows)
icc_stats_df['p_bh'] = multi.multipletests(icc_stats_df['p'].values, method='fdr_bh')[1]
icc_stats_df.to_csv(f'{indidiff_results_dir}/icc_pairwise_stats.csv', index=False)

def fmt_bf(v):
    v = float(v)
    return r'$\infty$' if not np.isfinite(v) else misc.write_scientific(v)

print('-' * 100)
print(f'Pairwise comparisons ({len(icc_stats_df)} pairs):')
print(f'  min $p_{{FDR}}$ = {misc.write_scientific(icc_stats_df["p_bh"].min())}, '
      f'max $p_{{FDR}}$ = {misc.write_scientific(icc_stats_df["p_bh"].max())}')
print(f'  min $BF_{{10}}$ = {fmt_bf(icc_stats_df["bf"].min())}, '
      f'max $BF_{{10}}$ = {fmt_bf(icc_stats_df["bf"].max())}')


# %%
# write top ICC (nodes and edges)

for atom_name in misc.atom_order:
    icc = np.load(f'{indidiff_results_dir}/icc_{atom_name}.npy')
    nodal_icc = np.nanmean(icc, axis=0)

    #print('-' * 100)
    #print(f'Top ICC nodes for {atom_name}')
    #print('-' * 100)
    #misc.print_topography_extremes(nodal_icc, n=50, parc=parcellation, expand=True)

    print('-' * 100)
    print(f'Top ICC edges for {atom_name}')
    print('-' * 100)
    misc.print_edge_extremes(icc, n=50, parc=parcellation, expand=True)


# %%

# ami of subject level community partitions to group
community_df = pd.read_csv(f'{networks_dir}/community.csv')
ami_df = community_df[community_df['measure'] == 'ami_vs_group'].copy()
ami_df = ami_df.rename(columns={'info': 'atom', 'value': 'ami'})

bp_kw = dict(
    showcaps=False, showfliers=False, patch_artist=True, widths=0.6,
    boxprops=dict(facecolor='none', edgecolor='black', linewidth=1, zorder=10),
    medianprops=dict(color='black', linewidth=1, zorder=11),
    whiskerprops=dict(color='black', linewidth=1, zorder=11),
)

# plot
np.random.seed(42)
fig, ax = plt.subplots(figsize=(3.5, 2.5))
for i, atom_name in enumerate(misc.atom_order):
    color = misc.atoms_colors[atom_name]
    vals = ami_df[ami_df['atom'] == atom_name]['ami'].values
    ax.scatter(
        np.full(len(vals), i) + np.random.normal(0, 0.06, len(vals)),
        vals, color=color, alpha=0.6, s=15, zorder=5,
    )
    ax.boxplot(vals, positions=[i], **bp_kw)

ax.set_xticks(range(len(misc.atom_order)))
ax.set_xticklabels([misc.atoms_names[a] for a in misc.atom_order], rotation=90)
for tick_label in ax.get_xticklabels():
    atom_key = [k for k, v in misc.atoms_names.items() if v == tick_label.get_text()]
    if atom_key:
        tick_label.set_color(misc.atoms_colors[atom_key[0]])
ax.set_xlim(-0.5, len(misc.atom_order) - 0.5)
ax.set_ylabel('Sim. to group')
sns.despine(ax=ax)
plt.tight_layout()
# fig.savefig(f'{plots_dir}/ami_vs_group.png', bbox_inches='tight')
plt.show()


# %%

# matrices of coassignment rates of nodes to communities across subjects
from matplotlib import patches

from palettable.scientific.diverging import Roma_5_r
cmap = Roma_5_r.mpl_colormap

_, communities = np.unique(atlas_info['network'].values, return_inverse=True)
sort_idx = np.argsort(communities)
sorted_communities = communities[sort_idx]
boundaries = np.where(np.diff(sorted_communities) != 0)[0] + 1
all_boundaries = [0] + boundaries.tolist() + [len(communities)]
unique_comms = np.unique(communities)
comm_colors = misc.distinguishable_colors(len(unique_comms))
color_map = dict(zip(unique_comms, comm_colors))

N = len(communities)
bar_width = max(1, N * 0.05)
gap = max(0.5, N * 0.02)

fig, axes = plt.subplots(2, 5, figsize=(22, 8))

for i, atom_name in enumerate(misc.atom_order):
    coassign = np.load(f'{indidiff_results_dir}/community_prob_{atom_name}.npy')
    sorted_mat = coassign[sort_idx, :][:, sort_idx]

    ax = axes[i // 5, i % 5]
    im = ax.imshow(sorted_mat, cmap=cmap, vmin=0, vmax=.6, aspect='auto')
    ax.set_title(misc.atoms_names[atom_name], color=misc.atoms_colors[atom_name])
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    for j in range(len(all_boundaries) - 1):
        start = all_boundaries[j]; end = all_boundaries[j + 1]
        color = color_map[sorted_communities[start]]
        rect_y = patches.Rectangle(
            (-0.5 - bar_width - gap, start - 0.5), bar_width, end - start,
            facecolor=color, edgecolor='black', clip_on=False,
        )
        ax.add_patch(rect_y)
        rect_x = patches.Rectangle(
            (start - 0.5, N - 0.5 + gap), end - start, bar_width,
            facecolor=color, edgecolor='black', clip_on=False,
        )
        ax.add_patch(rect_x)

plt.tight_layout()
# fig.savefig(f'{plots_dir}/coassignment_matrices.png', bbox_inches='tight')
plt.show()


# %%
# fingerprinting boxplots
import itertools
from scipy.stats import gaussian_kde

identi_df = pd.read_csv(f'{indidiff_results_dir}/identifiability.csv',
                        dtype={'subject': str})  # ids are zero-padded, keep them strings
identi_df = identi_df.rename(columns={'info': 'atom'})

metrics = [('iself', 'Self similarity'), ('iother', 'Other similarity'), ('idiff', 'Identifiability')]

# load idiff null distributions for permutation test and null overlay
idiff_nulls = {
    atom_name: np.load(f'{indidiff_results_dir}/idiff_null_{atom_name}.npy')
    for atom_name in misc.atom_order
}

# per-subject permutation p-values: compare each subject's observed idiff to their own null column
perm_pvals = {}
for atom_name in misc.atom_order:
    null = idiff_nulls[atom_name]  # (n_perm, n_subjects)
    n_perm = null.shape[0]
    atom_df = identi_df[identi_df['atom'] == atom_name].set_index('subject')
    obs = np.array([atom_df.loc[s, 'idiff'] for s in misc.subjects])  # (n_subjects,)
    p_sub = (np.sum(null >= obs[np.newaxis, :], axis=0) + 1) / (n_perm + 1)
    perm_pvals[atom_name] = p_sub  # (n_subjects,)

for metric, label in metrics:
    if metric == 'iself':
        fig, ax = plt.subplots(figsize=(3.25, 2.8))
    else:
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
    np.random.seed(42)
    for i, atom_name in enumerate(misc.atom_order):
        color = misc.atoms_colors[atom_name]
        vals = identi_df[identi_df['atom'] == atom_name][metric].values

        #if metric == 'idiff':
        #    null_pooled = idiff_nulls[atom_name].flatten()
        #    kde = gaussian_kde(null_pooled)
        #    y_range = np.linspace(null_pooled.min(), null_pooled.max(), 300)
        #    kde_vals = kde(y_range)
        #    kde_vals = kde_vals / kde_vals.max() * 0.35
        #    kde_vals[0] = 0
        #    kde_vals[-1] = 0
        #    ax.fill_betweenx(y_range, i - kde_vals, i,
        #                     facecolor='#FFFFFF', edgecolor='#898888',
        #                     linewidth=1, alpha=1, zorder=3)

        ax.scatter(
            np.full(len(vals), i) + np.random.normal(0, 0.06, len(vals)),
            vals, color=color, alpha=0.6, s=15, zorder=5,
        )
        ax.boxplot(vals, positions=[i],
                   showcaps=False, showfliers=False, patch_artist=True, widths=0.6,
                   boxprops=dict(facecolor='none', edgecolor='black', linewidth=1, zorder=10),
                   medianprops=dict(color='black', linewidth=1, zorder=11),
                   whiskerprops=dict(color='black', linewidth=1, zorder=11))

    if metric == 'idiff':
        trans = ax.get_xaxis_transform()
        for i, atom_name in enumerate(misc.atom_order):
            pct = np.mean(perm_pvals[atom_name] < 0.05)
            ax.text(i, 1.02, f'{int(pct * 100)}', ha='center', va='bottom',
                    rotation=90, color=misc.atoms_colors[atom_name],
                    transform=trans, clip_on=False)
        ax.text(-0.7, 1.1, '% sig.', ha='right', va='bottom',
                transform=trans, clip_on=False)

    ax.set_xticks(range(len(misc.atom_order)))
    ax.set_xticklabels([misc.atoms_names[a] for a in misc.atom_order], rotation=90)
    for tick_label in ax.get_xticklabels():
        atom_key = [k for k, v in misc.atoms_names.items() if v == tick_label.get_text()]
        if atom_key:
            tick_label.set_color(misc.atoms_colors[atom_key[0]])

    if metric == 'iself':
        ax.set_yticks([0.5, 0.8])
    
    ax.set_xlim(-0.5, len(misc.atom_order) - 0.5)
    ax.set_ylabel(label)
    sns.despine(ax=ax)
    plt.tight_layout()
    fig.savefig(f'{plots_dir}/fingerprinting_{metric}.png', bbox_inches='tight')
    plt.show()

# RM-ANOVA + pairwise paired t-tests for each metric
atom_pairs = list(itertools.combinations(misc.atom_order, 2))

anova_rows = []
pairwise_rows = []

for metric, label in metrics:
    df_long = identi_df[['subject', 'atom', metric]].copy()
    df_long = df_long.rename(columns={metric: 'value'})

    anova = pg.rm_anova(data=df_long, dv='value', within='atom', subject='subject')
    anova_rows.append({
        'metric': metric,
        'F': anova['F'].values[0],
        'ddof1': anova['ddof1'].values[0],
        'ddof2': anova['ddof2'].values[0],
        'p': anova['p_unc'].values[0],
        'ng2': anova['ng2'].values[0],
    })

    pair_stat_rows = []
    for atom_a, atom_b in atom_pairs:
        vals_a = identi_df[identi_df['atom'] == atom_a][metric].values
        vals_b = identi_df[identi_df['atom'] == atom_b][metric].values
        t, p, cil, ciu, d, bf, dof, power = misc.aug_t_test(vals_a, vals_b, paired=True)
        pair_stat_rows.append({
            'metric': metric, 'atom_a': atom_a, 'atom_b': atom_b,
            't': t, 'p': p, 'cil': cil, 'ciu': ciu,
            'd': d, 'bf': bf, 'dof': dof, 'power': power,
        })

    pair_df = pd.DataFrame(pair_stat_rows)
    _, pbh, _, _ = multi.multipletests(pair_df['p'].values, method='fdr_bh')
    pair_df['p_bh'] = pbh
    pairwise_rows.append(pair_df)

anova_stats = pd.DataFrame(anova_rows)
pairwise_stats = pd.concat(pairwise_rows, ignore_index=True)

anova_stats.to_csv(f'{indidiff_results_dir}/fingerprint_anova_stats.csv', index=False)
pairwise_stats.to_csv(f'{indidiff_results_dir}/fingerprint_pairwise_stats.csv', index=False)

# permutation p-values to CSV (per subject)
perm_rows = []
for atom_name in misc.atom_order:
    for s, p in zip(misc.subjects, perm_pvals[atom_name]):
        perm_rows.append({'atom': atom_name, 'subject': s, 'p_perm': p})
perm_df = pd.DataFrame(perm_rows)
perm_df.to_csv(f'{indidiff_results_dir}/fingerprint_perm_pvals.csv', index=False)

# print paper format
print('Idiff % significant participants:')
for atom_name in misc.atom_order:
    pct = np.mean(perm_pvals[atom_name] < 0.05) * 100
    print(f"{misc.atoms_names[atom_name]}: {pct:.0f}%")

print()
for _, row in anova_stats.iterrows():
    F_str = misc.write_scientific(row['F'])
    p_str = misc.write_scientific(row['p'])
    print(f"{row['metric']} ANOVA: $F$({int(row['ddof1'])},{int(row['ddof2'])})={F_str}, $p$={p_str}")

print()
print('Significant pairwise comparisons (p_FDR < 0.05):')
sig = pairwise_stats[pairwise_stats['p_bh'] < 0.05]
for _, row in sig.iterrows():
    t_str = misc.write_scientific(row['t'])
    p_str = misc.write_scientific(row['p_bh'])
    d_str = misc.write_scientific(row['d'])
    a = misc.atoms_names[row['atom_a']]
    b = misc.atoms_names[row['atom_b']]
    print(f"[{row['metric']}] {a} vs {b}: $t$={t_str}, $p_{{FDR}}$={p_str}, $d$={d_str}")


# %%
from scipy.stats import pearsonr
rc_fontsize = plt.rcParams['font.size']
ig_path = f'{indidiff_results_dir}/indiv_group_correlation.csv'
ig_method = 'pearson'
ig_div_variant = 'common_units'  # diversity in the original run's atom units

# the cache records which correlation produced it, so changing ig_method invalidates
# it instead of silently reloading the previous measure's numbers under a new name
_cached = pd.read_csv(ig_path) if os.path.exists(ig_path) else None
_usable = (_cached is not None and 'method' in _cached.columns
           and _cached['method'].iloc[0] == ig_method
           and 'div_variant' in _cached.columns
           and _cached['div_variant'].iloc[0] == ig_div_variant)

if _usable:
    ig_df = _cached
else:
    triu = np.triu_indices(n_regions, k=1)
    group_mats = np.load(f'{main_dir}/{parcellation}/group_phiid_matrix.npy',
                         allow_pickle=True).item()
    group_div = np.load(f'{main_dir}/{parcellation}/group_comp_diversity_normed_matrix.npy')
    # same per-atom rescaling as nndb2_phiid_s.ipynb, so individual and group
    # diversity are measured in the same units
    atom_anchor = (np.load(f'{main_dir}/{parcellation}/atom_rms_repl.npy')
                   / np.load(f'{main_dir}/{parcellation}/atom_rms_orig.npy'))
    group_grad = np.load(f'{main_dir}/{parcellation}/group_syn_red_grad.npy')
    indiv_grad = np.load(f'{main_dir}/{parcellation}/indiv_syn_red_grad.npy')

    group_vec = {a: group_mats[a][triu] for a in misc.atom_order}
    group_vec['Diversity'] = group_div[triu]

    _corr = {'pearson': pearsonr, 'spearman': spearmanr}[ig_method]

    def _rho(x, y):
        ok = np.isfinite(x) & np.isfinite(y)
        return _corr(x[ok], y[ok])[0]

    ig_rows = []
    for s_idx, perp in enumerate(tqdm(misc.subjects, desc='indiv vs group')):
        mat = np.load(f'{main_dir}/{parcellation}/sub-{perp}/phiid_matrix.npy'
                      ).astype(np.float64)
        for atom in misc.atom_order:
            v = mat[:, :, misc.atoms[atom]].sum(axis=2)[triu]
            ig_rows.append({'subject': perp, 'quantity': atom,
                            'r': _rho(v, group_vec[atom])})

        # computational diversity, recomputed as in nndb2_phiid_s.ipynb. the atom
        # loop above stays in raw units; only diversity needs the common-units scaling
        w = (mat / atom_anchor) ** 2
        s = w.sum(axis=2)
        with np.errstate(invalid='ignore', divide='ignore'):
            p = np.where(s[..., np.newaxis] != 0, w / s[..., np.newaxis], 1.0 / 16)
            div = np.where(s != 0, 1.0 / (np.sum(p ** 2, axis=2) * 16), 1.0)
        ig_rows.append({'subject': perp, 'quantity': 'Diversity',
                        'r': _rho(div[triu], group_vec['Diversity'])})

        # the gradient is a per-region array rather than a matrix
        ig_rows.append({'subject': perp, 'quantity': 'Gradient',
                        'r': _rho(indiv_grad[:, s_idx], group_grad)})
        del mat, w, s, p, div

    ig_df = pd.DataFrame(ig_rows)
    ig_df['method'] = ig_method
    ig_df['div_variant'] = ig_div_variant
    ig_df.to_csv(ig_path, index=False)

ig_quantities = misc.atom_order + ['Diversity', 'Gradient']
ig_labels = {**{a: misc.atoms_names[a] for a in misc.atom_order},
             'Diversity': 'Diversity ($D$)', 'Gradient': 'Red-Syn Grad'}
ig_colors = {**misc.atoms_colors, 'Diversity': '#6E6C6C', 'Gradient': '#B0AEAE'}

ncol = 4
nrow = int(np.ceil(len(ig_quantities) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 1.8 * nrow),
                         sharex=True, sharey=True)
axes = axes.flatten()
bins = np.linspace(0, 1, 50)

for ax, q in zip(axes, ig_quantities):
    vals = ig_df[ig_df['quantity'] == q]['r'].values
    sns.kdeplot(
        vals, 
        ax=ax, 
        color=ig_colors[q], 
        fill=False, alpha=1, linewidth=1.5,
        clip=(0, 1)
    )
    ax.axvline(np.median(vals), color='black', linestyle='--', linewidth=0.8)
    ax.set_title(ig_labels[q], color=ig_colors[q], fontsize=rc_fontsize, pad=4)
    #ax.text(0.03, 0.92, f'{np.median(vals):.2f}', transform=ax.transAxes,
    #        ha='left', va='top', fontsize=rc_fontsize - 4)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    sns.despine(ax=ax)

for ax in axes[len(ig_quantities):]:
    ax.set_visible(False)
for ax in axes[len(ig_quantities) - ncol:len(ig_quantities)]:
    ax.set_xlabel(r'Sim. to group')
for i, ax in enumerate(axes[:len(ig_quantities)]):
    if i % ncol == 0:
        ax.set_ylabel('Participants')

plt.tight_layout()
fig.savefig(f'{plots_dir}/indigroupcorr.png', bbox_inches='tight')
plt.show()

for q in ig_quantities:
    vals = ig_df[ig_df['quantity'] == q]['r'].values
    print(f'{ig_labels[q]:24s} sim. = {vals.mean():.3f} +/- {vals.std(ddof=1):.3f} '
          f'(range {vals.min():.3f} to {vals.max():.3f})')


# %%
comm_df = pd.read_csv(f'{networks_dir}/community.csv').rename(
    columns={'info': 'atom'})

ami_ind = comm_df[comm_df['measure'] == 'ami_vs_group'][['subject', 'atom', 'value']]
mod_ind = comm_df[comm_df['measure'] == 'modularity_individual'][['subject', 'atom', 'value']]
pc_ind = (comm_df[comm_df['measure'] == 'participation_coeff_individual']
          .groupby(['subject', 'atom'], as_index=False)['value'].mean())

comm_panels = [
    (ami_ind, 'Sim. to group'),
    (mod_ind, 'Modularity $Q$'),
    (pc_ind, 'Participation'),
]

bp_kw = dict(
    showcaps=False, showfliers=False, patch_artist=True, widths=0.6,
    boxprops=dict(facecolor='none', edgecolor='black', linewidth=1, zorder=10),
    medianprops=dict(color='black', linewidth=1, zorder=11),
    whiskerprops=dict(color='black', linewidth=1, zorder=11),
)

np.random.seed(42)
fig, axs = plt.subplots(1, len(comm_panels), figsize=(3.6 * len(comm_panels), 3.2))
for ax, (df, label) in zip(axs, comm_panels):
    for i, atom_name in enumerate(misc.atom_order):
        color = misc.atoms_colors[atom_name]
        vals = df[df['atom'] == atom_name]['value'].values
        ax.scatter(np.full(len(vals), i) + np.random.normal(0, 0.06, len(vals)),
                   vals, color=color, alpha=0.6, s=15, zorder=5)
        ax.boxplot(vals, positions=[i], **bp_kw)
    ax.set_ylabel(label)
    ax.set_xlim(-0.5, len(misc.atom_order) - 0.5)
    ax.set_xticks(range(len(misc.atom_order)))
    ax.set_xticklabels([misc.atoms_names[a] for a in misc.atom_order], rotation=90)
    for tick, atom_name in zip(ax.get_xticklabels(), misc.atom_order):
        tick.set_color(misc.atoms_colors[atom_name])
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    sns.despine(ax=ax)

plt.tight_layout()
fig.savefig(f'{plots_dir}/indigroup_comm.png', bbox_inches='tight')
plt.show()

# same contrast as Fig 2d, but on the individual partitions: each mode against the
# mean of the other nine, one-sample t on the per-participant difference
comm_rows = []
for df, label in comm_panels:
    piv = df.pivot(index='subject', columns='atom', values='value')[misc.atom_order]
    for atom in misc.atom_order:
        rest = [a for a in misc.atom_order if a != atom]
        diff = piv[atom].values - piv[rest].mean(axis=1).values
        t, p, cil, ciu, d, bf, dof, power = misc.aug_t_test(diff, np.zeros(len(diff)))
        comm_rows.append({'measure': label, 'atom': atom, 't': t, 'p': p,
                          'cil': cil, 'ciu': ciu, 'd': d, 'bf': float(bf),
                          'dof': dof, 'power': power,
                          'mean': float(piv[atom].mean()),
                          'sd': float(piv[atom].std(ddof=1))})
comm_stats = pd.DataFrame(comm_rows)
comm_stats['p_bh'] = multi.multipletests(comm_stats['p'].values, method='fdr_bh')[1]
comm_stats.to_csv(f'{networks_dir}/community_individual_atom_vs_rest.csv', index=False)

for df, label in comm_panels:
    piv = df.pivot(index='subject', columns='atom', values='value')[misc.atom_order]
    order = piv.mean().sort_values(ascending=False).index
    print(f'{label}: ' + ' > '.join(misc.atoms_texcommands[a] for a in order))
print()
print(f"significant at p_FDR < 0.05: "
      f"{int((comm_stats['p_bh'] < 0.05).sum())}/{len(comm_stats)}")
ami_vals = ami_ind['value'].values
print(f'AMI of individual to group partition: {ami_vals.mean():.3f} '
      f'+/- {ami_vals.std(ddof=1):.3f} (range {ami_vals.min():.3f} to {ami_vals.max():.3f})')


# %%
# writes the individual-differences statistics out as LaTeX tables for the supplementtt,
# table one is the per-mode summary behind Fig 3, table two the 45 pairwise identifiability
# contrasts. the full pairwise banks for iself, iother and ICC stay in the CSVs
tables_dir = f'{plots_dir}/tables'
os.makedirs(tables_dir, exist_ok=True)

identi = pd.read_csv(f'{indidiff_results_dir}/identifiability.csv',
                     dtype={'subject': str}).rename(columns={'info': 'atom'})
fp_anova = pd.read_csv(f'{indidiff_results_dir}/fingerprint_anova_stats.csv')
fp_pair = pd.read_csv(f'{indidiff_results_dir}/fingerprint_pairwise_stats.csv')
fp_perm = pd.read_csv(f'{indidiff_results_dir}/fingerprint_perm_pvals.csv')


def _sci(v):
    # misc.write_scientific leaves the mantissa outside math mode, which is fine
    # inline but not inside a tabular cell, so mirror it fully wrapped in $...$
    v = float(v)
    if not np.isfinite(v):
        return r'$\infty$'
    if v != 0 and (abs(v) < 1e-3 or abs(v) > 1e3):
        m, e = f'{v:.2e}'.split('e')
        return r'$%s\times10^{%d}$' % (m, int(e))
    return r'$%.3f$' % v


def _ms(vals):
    return r'$%.3f \pm %.3f$' % (float(np.mean(vals)), float(np.std(vals, ddof=1)))


# table 1: per-mode summary
summary_rows = []
for atom in misc.atom_order:
    icc = np.load(f'{indidiff_results_dir}/icc_{atom}.npy')
    triu = np.triu_indices(icc.shape[0], k=1)
    icc_vals = icc[triu]
    icc_vals = icc_vals[np.isfinite(icc_vals)]
    sub = identi[identi['atom'] == atom]
    pct = 100 * np.mean(fp_perm[fp_perm['atom'] == atom]['p_perm'].values < 0.05)
    summary_rows.append(r'%s & %s & %s & %s & %s & %.0f\%% \\' % (
        misc.atoms_texcommands[atom],
        _ms(icc_vals), _ms(sub['iself']), _ms(sub['iother']), _ms(sub['idiff']), pct))

anova_bits = []
for _, r in fp_anova.iterrows():
    lab = {'iself': r'$\mathrm{I_{self}}$', 'iother': r'$\mathrm{I_{other}}$',
           'idiff': 'identifiability'}[r['metric']]
    p_txt = (r'$p<10^{-300}$' if r['p'] < 1e-300
             else r'$p=%s$' % _sci(r['p']).strip('$'))
    anova_bits.append(r'%s $F(%d,%d)=%s$, %s, $\eta^2_G=%.3f$'
                      % (lab, int(r['ddof1']), int(r['ddof2']),
                         _sci(r['F']).strip('$'), p_txt, r['ng2']))

# table 2: pairwise identifiability
idiff_pair = fp_pair[fp_pair['metric'] == 'idiff']
pair_rows = []
for _, r in idiff_pair.iterrows():
    pair_rows.append(r'%s vs %s & %s & %.2f & %s & %s \\' % (
        misc.atoms_texcommands[r['atom_a']], misc.atoms_texcommands[r['atom_b']],
        _sci(r['t']), abs(float(r['d'])), _sci(r['bf']), _sci(r['p_bh'])))

tex = r"""%% generated by nndb2_indidiffs_s.ipynb, do not edit by hand
\begin{table*}[!ht]
\centering
%% thinner rules than the 0.4pt default, and \small confined to the tabular so
%% the caption keeps the normal body size
\begingroup
\setlength{\arrayrulewidth}{0.2pt}
\small
\begin{tabular}{lccccc}
\hline
Mode & ICC & $\mathrm{I_{self}}$ & $\mathrm{I_{other}}$ & Identifiability & \%% sig. \\
\hline
%(summary)s
\hline
\end{tabular}
\endgroup
\caption{\textbf{Reliability and identifiability of each mode of information dynamics.}}
\label{tab:indidiff}
\end{table*}

\begin{table*}[!ht]
\centering
\begingroup
\setlength{\arrayrulewidth}{0.2pt}
\small
\begin{tabular}{lcccc}
\hline
Contrast & $t$ & $|d|$ & $BF_{10}$ & $p_{\mathrm{FDR}}$ \\
\hline
%(pairs)s
\hline
\end{tabular}
\endgroup
\caption{\textbf{Pairwise contrasts of identifiability between modes of information dynamics.}
All %(npair)d pairs, by paired $t$-test across the $n=%(nsub)d$ participants
($\mathrm{df}=%(dof)d$), positive $t$ favouring the first mode of the pair. $p$-values are
FDR-corrected by the Benjamini-Hochberg procedure within this family of contrasts;
%(nsig)d of %(npair)d survive $p_{\mathrm{FDR}}<0.05$. Bayes factors are JZS Bayes factors under
a Cauchy prior of scale $r=0.707$.}
\label{tab:indidiff_pairs}
\end{table*}
""" % {'summary': '\n'.join(summary_rows), 'pairs': '\n'.join(pair_rows),
       'anova': '; '.join(anova_bits),
       'nsub': identi['subject'].nunique(),
       'nedge': f"{len(np.triu_indices(n_regions, k=1)[0]):,}".replace(',', '{,}'),
       'npair': len(idiff_pair), 'dof': int(idiff_pair['dof'].iloc[0]),
       'nsig': int((idiff_pair['p_bh'] < 0.05).sum())}

out = f'{tables_dir}/indidiff_table.tex'
with open(out, 'w') as fh:
    fh.write(tex)
print(f'wrote {out}: {len(summary_rows)} modes, {len(pair_rows)} pairwise contrasts')
print(f"identifiability pairs significant: {int((idiff_pair['p_bh'] < 0.05).sum())}/{len(idiff_pair)}")


# %%
# Nodal ICC maps tested gainst BrainSMASH surrogates
# the edge-level ICC matrices against a node-relabelling null.

orig_indidiff = f'../results/indidiffs/{parcellation}'
n_nulls = 1000

icc_csv = f'{indidiff_results_dir}/replication_icc.csv'
if os.path.exists(icc_csv):
    icc_rep = pd.read_csv(icc_csv)
else:
    rows = []
    for atom in tqdm(misc.atom_order, desc='ICC nulls'):
        o = np.load(f'{orig_indidiff}/icc_{atom}.npy')
        r = np.load(f'{indidiff_results_dir}/icc_{atom}.npy')
        m = misc.spatial_null_corr(np.nanmean(o, axis=0), np.nanmean(r, axis=0),
                                   n_nulls=n_nulls, seed=0)
        e = misc.matrix_null_corr(o, r, n_nulls=n_nulls, seed=0)
        rows.append({'atom': atom,
                     'r_nodal': m['r'], 'p_nodal': m['p'],
                     'r_edge': e['r'], 'p_edge': e['p'],
                     'mean_icc_orig': float(np.nanmean(o)),
                     'mean_icc_repl': float(np.nanmean(r))})
    icc_rep = pd.DataFrame(rows)
    for col in ('p_nodal', 'p_edge'):
        icc_rep[col + '_bh'] = multi.multipletests(icc_rep[col].values, method='fdr_bh')[1]
    icc_rep.to_csv(icc_csv, index=False)

for _, r in icc_rep.iterrows():
    print(f"{misc.atoms_names[r['atom']]:<16} nodal $r=${misc.write_scientific(r['r_nodal'])} "
          f"({misc.write_perm_p(r['p_nodal'], n_nulls)}), "
          f"edge $r=${misc.write_scientific(r['r_edge'])} "
          f"({misc.write_perm_p(r['p_edge'], n_nulls)})")

# the paper's claim is an ordering of modes by mean ICC, so test that too
icc_order = misc.perm_corr(icc_rep['mean_icc_orig'], icc_rep['mean_icc_repl'],
                           labels=list(icc_rep['atom']))
print(f"\nmode ordering by mean ICC: $r=${misc.write_scientific(icc_order['r'])}, "
      f"{misc.write_perm_p(icc_order['p'], icc_order['n_nulls'])}")
print('  NNDb:      ' + ' > '.join(
    icc_rep.sort_values('mean_icc_orig', ascending=False)['atom']))
print('  NNDb-3T+: ' + ' > '.join(
    icc_rep.sort_values('mean_icc_repl', ascending=False)['atom']))


# %%
# density panels for the ICC, original against replication. Nodal maps first
# (region-mean ICC, against the BrainSMASH surrogates), then the full edge sets
# (against the node-relabelling null); both p-values are the uncorrected ones
from scipy.ndimage import gaussian_filter
n_bins = 60
smooth = 1

cmap = plt.get_cmap('afmhot_r')

triu = np.triu_indices(n_regions, k=1)
icc_pairs = []
for atom in tqdm(misc.atom_order, desc='loading ICC'):
    o = np.load(f'{orig_indidiff}/icc_{atom}.npy')
    r = np.load(f'{indidiff_results_dir}/icc_{atom}.npy')
    icc_pairs.append((atom, np.nanmean(o, axis=0), np.nanmean(r, axis=0), o[triu], r[triu]))


def icc_density_panels(get_pair, r_col, p_col, fname):
    ncol = 5
    nrow = int(np.ceil(len(icc_pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.4 * nrow))
    axes = axes.flatten()

    for ax, entry in zip(axes, icc_pairs):
        atom = entry[0]
        a, b = get_pair(entry)
        keep = np.isfinite(a) & np.isfinite(b)
        a, b = a[keep], b[keep]

        counts, xe, ye = np.histogram2d(a, b, bins=n_bins)
        counts = gaussian_filter(counts, smooth)
        ax.imshow(counts.T, origin='lower', aspect='auto', cmap=cmap,
                  extent=[xe[0], xe[-1], ye[0], ye[-1]],
                  vmax=np.percentile(counts[counts > 0], 99),
                  interpolation='bilinear')

        row = icc_rep[icc_rep['atom'] == atom]
        r = row[r_col].values[0]
        p_val = row[p_col].values[0]
        ax.set_ylim(ye[0], ye[-1] + 0.26 * (ye[-1] - ye[0]))
        star = misc.get_star(p_val) if p_val < 0.05 else ''
        ax.text(0.06, 0.94, f'{star}', transform=ax.transAxes,
                ha='left', va='top', fontsize=plt.rcParams['font.size'] + 3, color='black')
        ax.text(0.98, 0.03, f'$r$ = {r:.2f}', transform=ax.transAxes,
                ha='right', va='bottom')
        ax.set_title(misc.atoms_names[atom], color='black',
                     fontsize=plt.rcParams['font.size'])
        ax.xaxis.set_major_locator(plt.MaxNLocator(2))
        ax.yaxis.set_major_locator(plt.MaxNLocator(2))
        sns.despine(ax=ax)

    for ax in axes[len(icc_pairs):]:
        ax.set_visible(False)
    for i, ax in enumerate(axes[:len(icc_pairs)]):
        if i % ncol == 0:
            ax.set_ylabel('Replication')
        if i >= len(icc_pairs) - ncol:
            ax.set_xlabel('Original')

    fig.align_ylabels(axes)

    plt.tight_layout()
    fig.savefig(f'{plots_dir}/{fname}', bbox_inches='tight')
    plt.show()


icc_density_panels(lambda e: (e[1], e[2]), 'r_nodal', 'p_nodal',
                   'replication_icc_nodal_corr.png')
icc_density_panels(lambda e: (e[3], e[4]), 'r_edge', 'p_edge',
                   'replication_icc_edge_corr.png')


# %%
# ttest whether modes keep relative ordering in fpringing

o_identi = pd.read_csv(f'{orig_indidiff}/identifiability.csv').rename(columns={'info': 'atom'})
r_identi = pd.read_csv(f'{indidiff_results_dir}/identifiability.csv',
                       dtype={'subject': str}).rename(columns={'info': 'atom'})
o_perm = pd.read_csv(f'{orig_indidiff}/fingerprint_perm_pvals.csv')
r_perm = pd.read_csv(f'{indidiff_results_dir}/fingerprint_perm_pvals.csv')

fp_rows = []
fp_vectors = {}  # kept for the scatter panels below
for metric, label in [('iself', 'Self similarity'), ('iother', 'Other similarity'),
                      ('idiff', 'Identifiability')]:
    a = np.array([o_identi[o_identi['atom'] == m][metric].mean() for m in misc.atom_order])
    b = np.array([r_identi[r_identi['atom'] == m][metric].mean() for m in misc.atom_order])
    fp_vectors[label] = (a, b)
    res = misc.perm_corr(a, b, labels=misc.atom_order)
    fp_rows.append({'metric': label, 'r': res['r'], 'p': res['p'],
                    'mean_orig': a.mean(), 'mean_repl': b.mean()})

fp_rep = pd.DataFrame(fp_rows)
fp_rep['p_bh'] = multi.multipletests(fp_rep['p'].values, method='fdr_bh')[1]
fp_rep.to_csv(f'{indidiff_results_dir}/replication_fingerprint.csv', index=False)

for _, r in fp_rep.iterrows():
    print(f"{r['metric']:<20} $r=${misc.write_scientific(r['r'])}, "
          f"{misc.write_perm_p(r['p'], 10000)} "
          f"(mean {r['mean_orig']:.3f} vs {r['mean_repl']:.3f})")

print('\n% of participants exceeding the identity-shuffling null:')
for m in misc.atom_order:
    po = 100 * np.mean(o_perm[o_perm['atom'] == m]['p_perm'].values < 0.05)
    pr = 100 * np.mean(r_perm[r_perm['atom'] == m]['p_perm'].values < 0.05)
    print(f'  {misc.atoms_names[m]:<16} {po:.0f}% vs {pr:.0f}%')


# %%
# per-mode profiles are ten points each, so these stay as scatters rather than
# densities, with every point coloured by its mode.
# perm_corr tests the ordering of the modes
from scipy.stats import rankdata

icc_prof = {'Mean ICC': (np.asarray(icc_rep['mean_icc_orig'], float),
                         np.asarray(icc_rep['mean_icc_repl'], float),
                         icc_order['r'], icc_order['p'])}
for _, row in fp_rep.iterrows():
    a, b = fp_vectors[row['metric']]
    icc_prof[row['metric']] = (a, b, row['r'], row['p'])

ncol = 4
nrow = int(np.ceil(len(icc_prof) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(2.6 * ncol, 2.4 * nrow))
axes = axes.flatten()

prof_labels = list(icc_prof)

for ax, label in zip(axes, prof_labels):
    a, b, r, p_val = icc_prof[label]
    ax.plot([0.5, 10.5], [0.5, 10.5], color='lightgrey', linestyle='--',
            linewidth=0.8, zorder=0)
    for atom, x, y in zip(misc.atom_order, rankdata(a), rankdata(b)):
        ax.scatter(x, y, s=28, color=misc.atoms_colors[atom],
                   edgecolor='black', linewidth=0.4, zorder=3)

    star = misc.get_star(p_val) if p_val < 0.05 else ''
    ax.text(0.06, 0.94, f'{star}', transform=ax.transAxes,
            ha='left', va='top', fontsize=plt.rcParams['font.size'] + 3, color='black')
    ax.text(0.98, 0.03, f'$r$ = {r:.2f}', transform=ax.transAxes,
            ha='right', va='bottom')
    ax.set_title(label, color='black', fontsize=plt.rcParams['font.size'])
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 12.4)  # ranks stop at 10; the strip above holds the label
    ax.set_xticks([1, 10])
    ax.set_yticks([1, 10])
    sns.despine(ax=ax)

for ax in axes[len(prof_labels):]:
    ax.set_visible(False)
for i, ax in enumerate(axes[:len(prof_labels)]):
    if i % ncol == 0:
        ax.set_ylabel('Replication (rank)')
    if i >= len(prof_labels) - ncol:
        ax.set_xlabel('Original (rank)')

fig.align_ylabels(axes)

plt.tight_layout()
fig.savefig(f'{plots_dir}/replication_iccprofile_corr.png', bbox_inches='tight')
plt.show()


# %%
# mkkay so iccs dont replicate that well, but self and other sim does, identifiability does but not perfect
