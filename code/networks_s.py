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
plt.rcParams.update({'font.size': 15})

import network_utils as nu
import misc

parcellation = misc.parcellation
n_regions = misc.n_regions

MRI_path = '/home/george/Research/fMRI_general'
atlas_info = pd.read_csv(f'{MRI_path}/atlases/schaefer_extend_atlases/{parcellation}.csv')

main_dir = f'results/main'

plots_dir = f'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

networks_dir = f'results/networks/{parcellation}'
if not os.path.exists(networks_dir):
    os.makedirs(networks_dir)

import sys
sys.path.append(f'{MRI_path}')
from cortexcanvas import (plot_wholebrain, plot_full, 
                          plot_wholebrain_row, plot_wholebrain_row_2,
                          plot_grand_ensemble, plot_brain_network)

from palettable.scientific.diverging import Roma_5_r, Vik_3, Berlin_20

def load_connectome(perp, atom_indices, fix_nans=True):
    mat = np.load(f'{main_dir}/{parcellation}/sub-{perp}/phiid_matrix.npy')
    out = mat[:, :, atom_indices].sum(axis=2).astype(np.float64)
    if fix_nans:
        if np.isnan(out).any():
            min_finite = np.nanmin(out)
            out = np.where(np.isnan(out), min_finite, out)
    return out


# %%
from matplotlib import patches

group_phiid_path = f'{main_dir}/{parcellation}/group_phiid_matrix.npy'
group_phiid_matrix = np.load(group_phiid_path, allow_pickle=True).item()

def load_yeo_labels(df):
    _, encoded = np.unique(df['network'].values, return_inverse=True)
    return encoded

communities = load_yeo_labels(atlas_info)

# indices that sort the nodes by community
sort_idx = np.argsort(communities)
sorted_communities = communities[sort_idx]
# boundaries where communities change
boundaries = np.where(np.diff(sorted_communities) != 0)[0] + 1
all_boundaries = [0] + boundaries.tolist() + [len(communities)]
# map each unique community to a specific color
unique_comms = np.unique(communities)
comm_colors = misc.distinguishable_colors(len(unique_comms))
color_map = dict(zip(unique_comms, comm_colors))

fig, axes = plt.subplots(2, 5, figsize=(22, 8))

for i, atom in enumerate(misc.atom_order):
    values = group_phiid_matrix[atom]
    
    reordered_values = values[sort_idx, :][:, sort_idx]
    
    ax = axes[i // 5, i % 5]
    im = ax.imshow(reordered_values, 
                   #cmap=misc.div_cmap,
                   cmap=Roma_5_r.mpl_colormap,
                   vmin=np.nanpercentile(reordered_values, 5), 
                   vmax=np.nanpercentile(reordered_values, 95))
    ax.set_title(f'{atom} ({misc.atoms_names[atom]})')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_xticks([])
    ax.set_yticks([])
    
    N = len(communities)
    bar_width = max(1, N * 0.05)
    gap = max(0.5, N * 0.02)
    for j in range(len(all_boundaries) - 1):
        start = all_boundaries[j]; end = all_boundaries[j+1]
        length = end - start
        comm_label = sorted_communities[start]
        color = color_map[comm_label]
        rect_y = patches.Rectangle((-0.5 - bar_width - gap, start - 0.5), bar_width, length, 
                                   facecolor=color, edgecolor='black', clip_on=False)
        ax.add_patch(rect_y)
        rect_x = patches.Rectangle((start - 0.5, N - 0.5 + gap), length, bar_width, 
                                   facecolor=color, edgecolor='black', clip_on=False)
        ax.add_patch(rect_x)

plt.tight_layout() 
plt.show()


# %%
#SC
s_conn = loadmat(f'{MRI_path}/DTI/HCP1065_DTI_{parcellation}.mat')['count_size_length_norm_end']
s_length = loadmat(f'{MRI_path}/DTI/HCP1065_DTI_{parcellation}.mat')['mean_length_end']

# disconnected nodes?
degree = np.sum(s_conn > 0, axis=0)
n_disconnected = np.sum(degree == 0)
print(f'Number of disconnected nodes: {n_disconnected}')

#ED
from scipy.spatial import distance
coords = atlas_info[['x', 'y', 'z']].values[:n_regions, :]
s_dist = distance.pdist(coords)
s_dist = distance.squareform(s_dist)

np.save(f'{networks_dir}/ED.npy', s_dist)
np.save(f'{networks_dir}/SC.npy', s_conn)

cmap1 = 'binary'
from palettable.scientific.diverging import Roma_5_r
cmap2 = Roma_5_r.mpl_colormap

fig, axs = plt.subplots(1, 3, figsize=(3     * 5, 5))
vmin = np.min(s_conn); vmax = np.percentile(s_conn, 97.5)
print(f"Structural Connectivity - vmin: {vmin}, vmax: {vmax}")
axs[0].imshow(s_conn, cmap=cmap1, vmin=vmin, vmax=vmax)
axs[0].set_title('Structural Connectivity')
axs[0].set_xticks([]); axs[0].set_yticks([])
vmin = np.min(s_length); vmax = np.percentile(s_length, 97.5)
print(f"Mean Length - vmin: {vmin}, vmax: {vmax}")
axs[1].imshow(s_length, cmap=cmap1, vmin=vmin, vmax=vmax)
axs[1].set_title('Tract Length')
axs[1].set_xticks([]); axs[1].set_yticks([])
vmin = np.min(s_dist); vmax = np.percentile(s_dist, 97.5)
print(f"Distance - vmin: {vmin}, vmax: {vmax}")
axs[2].imshow(s_dist, cmap=cmap2, vmin=vmin, vmax=vmax)
axs[2].set_title('Euclidean Distance')
axs[2].set_xticks([]); axs[2].set_yticks([])
plt.tight_layout()

corr, p = spearmanr(s_conn[np.triu_indices_from(s_conn, k=1)], s_dist[np.triu_indices_from(s_dist, k=1)])
print(f"Correlation between SC and ED: r = {corr:.4f}, p-value = {p:.4e}")


# %%
kwargs = dict(node_color="#CCC8C8", edge_color="#2A2A2A", node_edgecolor="#3A3939", node_linewidth=.5,
              edge_alpha=(0, 1),  edge_width=(0.5, 5.0), node_size=(1, 300))
p99 = np.percentile(s_conn[s_conn > 0], 99.9)
s_conn_clipped = np.clip(s_conn, 0, p99)
fig, ax = plot_brain_network(s_conn_clipped, coords=atlas_info[['x', 'y', 'z']].values[:n_regions, :],
                             edge_scores=s_conn_clipped, node_scores=np.sum(s_conn_clipped, axis=0),
                             **kwargs)
fig.savefig(f'{plots_dir}/sc_brain_network.png', bbox_inches='tight')
fig, ax = plot_brain_network(s_conn_clipped, coords=atlas_info[['x', 'y', 'z']].values[:n_regions, :],
                             edge_scores=s_conn_clipped, node_scores=np.sum(s_conn_clipped, axis=0), view='side',
                             **kwargs)
plt.tight_layout()


# %%
group_phiid_path = f'{main_dir}/{parcellation}/group_phiid_matrix.npy'
group_phiid_matrix = np.load(group_phiid_path, allow_pickle=True).item()


def _top_edges(W, n_keep=1000):
    triu = np.triu_indices(W.shape[0], k=1)
    W = np.where(np.isnan(W), np.nanmin(W), W)
    ranked = nu.rank_normalise(W)
    utri_vals = ranked[triu]
    top_idx = np.argsort(utri_vals)[::-1][:n_keep]
    out = np.zeros_like(W, dtype=float)
    r, c = triu[0][top_idx], triu[1][top_idx]
    out[r, c] = utri_vals[top_idx]
    out[c, r] = utri_vals[top_idx]
    return out


for i, atom in enumerate(misc.atom_order):
    matrix = _top_edges(group_phiid_matrix[atom], n_keep=1000)

    kwargs_atom = kwargs.copy()
    kwargs_atom['node_color'] = misc.atoms_colors[atom]
    kwargs_atom['edge_width'] = (0.1, 3.0)

    fig, ax = plot_brain_network(matrix, coords=atlas_info[['x', 'y', 'z']].values[:n_regions, :],
                                 edge_scores=matrix, node_scores=np.sum(matrix, axis=0),
                                 **kwargs_atom)
    fig.savefig(f'{plots_dir}/phiid_{atom}_network.png', bbox_inches='tight')
    plt.close()


# %%
import network_utils as nu

comm_dict = {'SP': 'Shortest Path Efficiency',
            'NE': 'Navigation Efficiency',
            'SI': 'Search Information',
            'CE': 'Communicability',
            'DE': 'Diffusion Efficiency'}
comm_names = {'SP': 'Routing',
            'NE': 'Navigation',
            'SI': 'Search',
            'CE': 'Broadcast',
            'DE': 'Diffusion'}

comms_use = ['SP', 'NE', 'CE', 'DE']

if not os.path.exists(f'{networks_dir}/CM.npy'):
    models = nu.compute_communication_models(s_conn, s_dist, invert_SI=True)
    np.save(f'{networks_dir}/CM.npy', models)
else:
    models = np.load(f'{networks_dir}/CM.npy', allow_pickle=True).item()


# %%
n_models = len(comms_use)
fig, axs = plt.subplots(1, n_models, figsize=(n_models * 4, 4))

for i, model in enumerate(comms_use):
    ax = axs[i]
    matrix = models[model]

    # plot with SC
    combined = np.tril(s_conn, k=-1) + np.triu(matrix)
    mask_upper = np.triu(np.ones_like(combined, dtype=bool))
    mask_lower = np.tril(np.ones_like(combined, dtype=bool), k=-1)
    m_upper = np.ma.masked_where(~mask_upper, combined)
    m_lower = np.ma.masked_where(~mask_lower, combined)
    upper_values = combined[mask_upper]
    lower_values = combined[mask_lower]
    ax.imshow(m_upper, cmap=cmap2, 
             vmin=np.percentile(upper_values, 5), 
             vmax=np.percentile(upper_values, 95))
    ax.imshow(m_lower, cmap=cmap1,
             vmin=np.percentile(lower_values, 5), 
             vmax=np.percentile(lower_values, 97.5))
    if i == 0:
        ax.set_ylabel('Structural Connectivity', fontsize=17)
    
    # plot as is
    #ax.imshow(matrix, cmap=cmap2, vmin=np.percentile(matrix, 5), vmax=np.percentile(matrix, 95))

    ax.set_title(comm_names[model], fontsize=15)
    ax.set_xticks([])
    ax.set_yticks([])

    #ax.axhline(int((n_regions-54)/2), color='black', lw=1)
    #ax.axvline(int((n_regions-54)/2), color='black', lw=1)
    #ax.axhline(int(n_regions-54), color='black', lw=1)
    #ax.axvline(int(n_regions-54), color='black', lw=1)

fig.tight_layout()
# five 1054x1054 matrices at the notebook default of 800 dpi is a 25 MB file for
# no visible gain, since the panels are already far below one pixel per edge
fig.savefig(f'{plots_dir}/CM_matrices.png', bbox_inches='tight', dpi=300)


# %%
n_bins = 100

if not os.path.exists(f'{networks_dir}/strength_by_distance.npy'):

    # load ED and rank-normalise
    ed = np.load(f'{networks_dir}/ED.npy')
    ed_ranked = nu.rank_normalise(ed)

    # upper triangle indices (excluding diagonal)
    triu_idx = np.triu_indices(ed.shape[0], k=1)
    ed_upper = ed_ranked[triu_idx]

    # 100 equal-width bins in rank space
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(ed_upper, bin_edges[1:-1])
    bin_counts = np.bincount(bin_ids, minlength=n_bins)

    # per-subject, per-atom mean rank in each distance bin
    n_atoms = len(misc.atom_order)
    all_means = np.zeros((len(misc.subjects), n_atoms, n_bins))

    for s_idx, sub_id in enumerate(tqdm(misc.subjects, desc='subjects')):
        for a_idx, atom_name in enumerate(misc.atom_order):
            atom_mat = load_connectome(sub_id, misc.atoms[atom_name])
            atom_upper = nu.rank_normalise(atom_mat)[triu_idx]
            bin_sums = np.bincount(bin_ids, weights=atom_upper, minlength=n_bins)
            all_means[s_idx, a_idx] = bin_sums / np.maximum(bin_counts, 1)

    # mean and SE across participants
    mean_subs = np.mean(all_means, axis=0)
    se_subs = np.std(all_means, axis=0) / np.sqrt(all_means.shape[0])
    strength_by_distance = {'mean': mean_subs, 'se': se_subs, 'all': all_means}
    np.save(f'{networks_dir}/strength_by_distance.npy', strength_by_distance)
else:
    strength_by_distance = np.load(f'{networks_dir}/strength_by_distance.npy', allow_pickle=True).item()
    mean_subs = strength_by_distance['mean']
    se_subs = strength_by_distance['se']

x = np.arange(n_bins) / n_bins

fig, ax = plt.subplots(figsize=(6.5, 3.55))

for a_idx, atom_name in enumerate(misc.atom_order):
    color = misc.atoms_colors[atom_name]
    label = misc.atoms_names[atom_name]
    mean = mean_subs[a_idx]
    se = se_subs[a_idx]
    ax.plot(x, mean, color=color, label=label, linewidth=1.5)
    ax.fill_between(x, mean - se, mean + se, color=color, alpha=0.2, linewidth=0)

ax.set_xlabel('Geometric distance')
ax.set_ylabel('Strength')
leg = ax.legend(title='', bbox_to_anchor=(1.05, 1.05), loc='upper left', frameon=False,
                handlelength=1, handletextpad=0.5, labelspacing=0.3)
for line in leg.get_lines():
    line.set_linewidth(2)
ax.set_xticks([0, 0.5, 1])
ax.yaxis.set_major_locator(plt.MaxNLocator(2))
ax.set_xlim(None, 1)
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks([0, 0.5, 1])
ax2.set_xticklabels(['short-range', 'mid-range', 'long-range'])
sns.despine(ax=ax); sns.despine(ax=ax2)

#ax.axvspan(0, 0.33, color='grey', alpha=0.1, zorder=-1)
#ax.axvspan(0.33, 0.66, color='grey', alpha=0.05, zorder=-1)
#ax.axvspan(0.66, 1, color='grey', alpha=0.1, zorder=-1)
plt.tight_layout()
fig.savefig(f'{plots_dir}/strength_by_distance.png', bbox_inches='tight')


# %%
# per-subject quadratic fit of strength vs rank-distance: strength ~ b0 + b1*x + b2*x^2
sbd = np.load(f'{networks_dir}/strength_by_distance.npy', allow_pickle=True).item()
all_means = sbd['all']
n_sub, n_atoms, n_bins = all_means.shape
x = (np.arange(n_bins) + 0.5) / n_bins
xc = x - x.mean()
V = np.polynomial.polynomial.polyvander(xc, 2)   # columns [1, x, x^2]

rows = []
for a_idx, atom in enumerate(misc.atom_order):
    coefs = np.array([np.linalg.lstsq(V, all_means[s, a_idx], rcond=None)[0]
                      for s in range(n_sub)])       # (n_sub, 3): b0, b1, b2
    b1, b2 = coefs[:, 1], coefs[:, 2]
    _, p_lin, *_  = misc.aug_t_test(b1, np.zeros(n_sub))
    _, p_quad, *_ = misc.aug_t_test(b2, np.zeros(n_sub))
    vertex = np.nanmean(x.mean() - b1 / (2 * b2))
    rows.append({'atom': atom, 'b1': b1.mean(), 'b2': b2.mean(),
                 'p_lin': p_lin, 'p_quad': p_quad, 'vertex': vertex})
fit_df = pd.DataFrame(rows)
for c in ['p_lin', 'p_quad']:
    fit_df[c + '_bh'] = multi.multipletests(fit_df[c].values, method='fdr_bh')[1]

for _, r in fit_df.iterrows():
    interior = 0 <= r['vertex'] <= 1
    shape = ('inverted-U' if r['b2'] < 0 else 'U-shaped') if interior else 'monotonic'
    print(fr"{misc.atoms_names[r['atom']]}: "
          fr"$\beta_1 = ${misc.write_scientific(r['b1'])} "
          fr"($p_{{\mathrm{{FDR}}}} = ${misc.write_scientific(r['p_lin_bh'])}), "
          fr"$\beta_2 = ${misc.write_scientific(r['b2'])} "
          fr"($p_{{\mathrm{{FDR}}}} = ${misc.write_scientific(r['p_quad_bh'])}), "
          fr"vertex $= {r['vertex']:.2f}$ ({shape})")


# %%
# per-participant rank-biserial correlation: direct vs indirect


if not os.path.exists(f'{networks_dir}/direct_indirect_rrb.csv'):
    sc = np.load(f'{networks_dir}/SC.npy')
    np.fill_diagonal(sc, 0)
    triu_idx = np.triu_indices(sc.shape[0], k=1)
    sc_utri = sc[triu_idx]
    direct_mask = sc_utri > 0
    indirect_mask = ~direct_mask

    rows = []
    for sub_id in tqdm(misc.subjects, desc='subjects'):
        for atom_name in misc.atom_order:
            atom_mat = load_connectome(sub_id, misc.atoms[atom_name])
            atom_utri = atom_mat[triu_idx]
            _, _, rbc, _ = misc.aug_mwu_test(atom_utri[direct_mask], atom_utri[indirect_mask])
            rows.append({'subject': sub_id, 'atom': atom_name, 'rbc': rbc})

    rrb_df = pd.DataFrame(rows)
    rrb_df.to_csv(f'{networks_dir}/direct_indirect_rrb.csv', index=False)
else:
    rrb_df = pd.read_csv(f'{networks_dir}/direct_indirect_rrb.csv')

# one-sample t-test against zero, BH correction
stat_rows = []
for atom_name in misc.atom_order:
    vals = rrb_df[rrb_df['atom'] == atom_name]['rbc'].values
    t, p, cil, ciu, cohen_d, bf, dof, power = misc.aug_t_test(vals, np.zeros(len(vals)))
    stat_rows.append({'atom': atom_name, 't': t, 'p': p, 'cil': cil, 'ciu': ciu,
                      'd': cohen_d, 'bf': float(bf), 'dof': dof, 'power': power})
rrb_stats = pd.DataFrame(stat_rows)
_, pvals_bh, _, _ = multi.multipletests(rrb_stats['p'].values, method='fdr_bh')
rrb_stats['p_bh'] = pvals_bh

for atom_name in misc.atom_order:
    row = rrb_stats[rrb_stats['atom'] == atom_name].iloc[0]
    print(f"{atom_name}: $T=${misc.write_scientific(row['t'])}, "
          f"$p_{{\\mathrm{{FDR}}}}=${misc.write_scientific(row['p_bh'])}, "
          f"$d=${misc.write_scientific(row['d'])}, "
          f"$BF_{{10}}=${misc.write_scientific(row['bf'])}")
# print max p and min bf
print(rf"Max $p_{{\mathrm{{FDR}}}}$ = {misc.write_scientific(rrb_stats['p_bh'].max())}, "
      rf"Min $BF_{{10}}$ = {misc.write_scientific(rrb_stats['bf'].min())}")

np.random.seed(42)
fig, ax = plt.subplots(figsize=(3.7, 3.25))

bp_kw = dict(showcaps=False, showfliers=False, patch_artist=True, widths=0.6,
             boxprops=dict(facecolor='none', edgecolor='black', linewidth=1, zorder=10),
             medianprops=dict(color='black', linewidth=1, zorder=11),
             whiskerprops=dict(color='black', linewidth=1, zorder=11))

for i, atom_name in enumerate(misc.atom_order):
    color = misc.atoms_colors[atom_name]
    vals = rrb_df[rrb_df['atom'] == atom_name]['rbc'].values
    ax.scatter(np.full(len(vals), i) + np.random.normal(0, 0.06, len(vals)), vals,
               color=color, alpha=0.6, s=15, zorder=5)
    ax.boxplot(vals, positions=[i], **bp_kw)

ax.axhline(0, color='grey', linestyle='--', linewidth=0.8, zorder=0)

y0, y1 = ax.get_ylim()
yr = y1 - y0
ax.set_ylim(y0, y1 + yr * 0.12)

for i, atom_name in enumerate(misc.atom_order):
    row = rrb_stats[rrb_stats['atom'] == atom_name].iloc[0]
    if row['p_bh'] < 0.05:
        ax.text(i, y1 + yr * 0.01, misc.get_star(row['p_bh']),
                ha='center', va='bottom', fontsize=14)

ax.set_xticks(range(len(misc.atom_order)))
ax.set_xticklabels([misc.atoms_names[a] for a in misc.atom_order], rotation=90)
#set colors of xtick labels to match atom colors
for tick_label in ax.get_xticklabels():
    atom_name = tick_label.get_text()
    atom_key = [k for k, v in misc.atoms_names.items() if v == atom_name][0]
    tick_label.set_color(misc.atoms_colors[atom_key])
ax.set_yticks([-0.3, 0, 0.3])
y_lim = max(abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1]))
ax.set_ylim(-y_lim, y_lim)
ax.yaxis.set_label_position("right")
ax.yaxis.tick_right()
ax.text(-1, 0.25, 'Direct', ha='center', va='center', rotation=90, color='black')
ax.text(-1, -0.25, 'Indirect', ha='center', va='center', rotation=90, color='grey')
ax.set_xlim(-0.5, len(misc.atom_order) - 0.5)
sns.despine(ax=ax, left=True)
plt.tight_layout()
fig.savefig(f'{plots_dir}/direct_indirect_rbc.png', bbox_inches='tight')


# %%
# Communication models

r2_df = pd.read_csv(f'{networks_dir}/matrix_r2.csv')
# search info redundant with communicability
r2_df = r2_df[~((r2_df['predictor'] == 'CM_SI') & (r2_df['info'].isin(misc.atom_order)))]
r2_df = r2_df.rename(columns={'info': 'atom'})

cm_order = ['CM_SP', 'CM_NE', 'CM_CE', 'CM_DE']
cm_labels = [comm_names[cm.replace('CM_', '')] for cm in cm_order]
n_atoms = len(misc.atom_order)
n_cm = len(cm_order)
group_width = n_cm

# mean R2 and fraction of participants significant, per atom and communication model
for cm in cm_order:
    mean_r2_list, frac_sig_list, p_perm_list = [], [], []
    model = comm_names[cm.replace('CM_', '')]
    for atom in misc.atom_order:
        sub = r2_df[(r2_df['atom'] == atom) & (r2_df['predictor'] == cm)]
        mean_r2 = np.nanmean(sub['r2'].values)
        frac_sig = np.nanmean(sub['p_perm'].values < 0.05)
        p_perm = np.nanmean(sub['p_perm'].values)
        mean_r2_list.append(mean_r2)
        frac_sig_list.append(frac_sig)
        p_perm_list.append(sub['p_perm'].values)
        print(fr"{misc.atoms_names[atom]} - {model}: "
              fr"$\bar{{R^2}}$ = {misc.write_scientific(mean_r2)}, "
              fr"$\bar{{p}}_{{\mathrm{{perm}}}}$ = {misc.write_scientific(p_perm)}, "
              fr"\% sig = {frac_sig * 100:.0f}\%")
    print(fr"{model}: mean $\bar{{R^2}}$ = {misc.write_scientific(np.nanmean(mean_r2_list))}, "
          fr"mean $\bar{{p}}_{{\mathrm{{perm}}}}$ = {misc.write_scientific(np.nanmean(p_perm_list))}, "
          fr"mean \% sig = {misc.write_scientific(np.nanmean(frac_sig_list) * 100)}\%")
          

fig, ax = plt.subplots(figsize=(10, 4.25))

bp_kw = dict(showcaps=False, showfliers=False, patch_artist=True, widths=0.6,
             boxprops=dict(facecolor='none', edgecolor='black', linewidth=1, zorder=10),
             medianprops=dict(color='black', linewidth=1, zorder=11),
             whiskerprops=dict(color='black', linewidth=1, zorder=11))

xtick_pos, xtick_labels, pct_data = [], [], []

for i, atom in enumerate(misc.atom_order):
    color = misc.atoms_colors[atom]
    for j, (cm, cm_lbl) in enumerate(zip(cm_order, cm_labels)):
        x = i * group_width + j
        vals = r2_df[(r2_df['atom'] == atom) & (r2_df['predictor'] == cm)]['r2'].values
        pvals_sub = r2_df[(r2_df['atom'] == atom) & (r2_df['predictor'] == cm)]['p_perm'].values
        ax.scatter(x + np.random.normal(0, 0.12, len(vals)), vals,
                   color=color, alpha=0.5, s=12, zorder=5)
        ax.boxplot(vals, positions=[x], **bp_kw)
        xtick_pos.append(x)
        xtick_labels.append(cm_lbl if i == 0 else '')
        pct_data.append((x, float(np.mean(pvals_sub < 0.05)), color))
    if i < n_atoms - 1:
        # extend vertical line higher above plot
        ax.axvline(i * group_width + n_cm - 0.5, color='grey', linestyle=':',
                   linewidth=1.5, zorder=0)

ax.set_yscale('log')
ax.set_xticks(xtick_pos)
ax.set_xticklabels(xtick_labels, rotation=90)
ax.set_ylabel('$R^2$')
ax.set_xlim(-0.7, (n_atoms - 1) * group_width + n_cm - 0.3)
sns.despine(ax=ax)

trans = ax.get_xaxis_transform()

for x, pct, color in pct_data:
    ax.text(x, 1.02, f'{int(pct * 100)}', ha='center', va='bottom',
            rotation=90, color=color, transform=trans, clip_on=False)
ax.text(-1.2, 1.1, '% sig.', ha='right', va='bottom',
        transform=trans, clip_on=False)

for i, atom in enumerate(misc.atom_order):
    x_center = i * group_width + (n_cm - 1) / 2
    ax.text(x_center, 1.3, misc.atoms_names[atom], ha='center', va='bottom',
            color=misc.atoms_colors[atom], transform=trans, clip_on=False)
plt.tight_layout()

fig.savefig(f'{plots_dir}/communication_models_r2.png', bbox_inches='tight')

plt.show()


# %%
comm_df = pd.read_csv(f'{networks_dir}/community.csv')
comm_df = comm_df.rename(columns={'info': 'atom'})

def mean_pc(partition):
    return comm_df[comm_df['measure'] == f'participation_coeff_{partition}'].groupby(
        ['subject', 'atom'], as_index=False)['value'].mean()

pc_ind = mean_pc('individual')
pc_grp = mean_pc('group')

mod_ind = comm_df[comm_df['measure'] == 'modularity_individual'][['subject', 'atom', 'value']]
mod_grp = comm_df[comm_df['measure'] == 'modularity_group'][['subject', 'atom', 'value']]

ami_yeo = comm_df[comm_df['measure'] == 'ami_vs_yeo'][['subject', 'atom', 'value']]

topo_df = pd.read_csv(f'{networks_dir}/topology_global.csv')
topo_df = topo_df.rename(columns={'info': 'atom'})

bp_kw = dict(showcaps=False, showfliers=False, patch_artist=True, widths=0.6,
             boxprops=dict(facecolor='none', edgecolor='black', linewidth=1, zorder=10),
             medianprops=dict(color='black', linewidth=1, zorder=11),
             whiskerprops=dict(color='black', linewidth=1, zorder=11))


# %%
panels = [
    (topo_df[topo_df['measure'] == 'global_efficiency'],  'Efficiency'),
    (topo_df[topo_df['measure'] == 'von_neumann_entropy'], 'Complexity'),
    (topo_df[topo_df['measure'] == 's_metric'], 'Hubness'),
    (mod_grp, 'Modularity'),
    (pc_grp, 'Participation'),
    (ami_yeo, 'Sim. to Yeo'),
]

n_panels = len(panels)
np.random.seed(42)

fig, axs = plt.subplots(n_panels, 1, figsize=(3.8, 1.6 * n_panels), sharex=True)

for ax_idx, (df, label) in enumerate(panels):
    ax = axs[ax_idx]
    if label == 'Hubness':
        raw = df['value'].values
        lo, hi = raw.min(), raw.max()
        scale = lambda v: (v - lo) / (hi - lo)
    else:
        scale = lambda v: v

    for i, atom_name in enumerate(misc.atom_order):
        color = misc.atoms_colors[atom_name]
        vals = scale(df[df['atom'] == atom_name]['value'].values)
        ax.scatter(np.full(len(vals), i) + np.random.normal(0, 0.06, len(vals)),
                   vals, color=color, alpha=0.6, s=15, zorder=5)
        ax.boxplot(vals, positions=[i], **bp_kw)
    ax.set_ylabel(label)
    ax.set_xlim(-0.5, len(misc.atom_order) - 0.5)
    ax.yaxis.set_major_locator(plt.LinearLocator(2))
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'{v:.2f}'.rstrip('0').rstrip('.'))
    )
    sns.despine(ax=ax)



axs[-1].set_xticks(range(len(misc.atom_order)))
axs[-1].set_xticklabels([misc.atoms_names[a] for a in misc.atom_order], rotation=90)
for tick, atom_name in zip(axs[-1].get_xticklabels(), misc.atom_order):
    tick.set_color(misc.atoms_colors[atom_name])

fig.align_ylabels(axs)
plt.tight_layout()
fig.savefig(f'{plots_dir}/topology_panels.png', bbox_inches='tight')


# %%
topo = pd.read_csv(f'{networks_dir}/topology_global.csv').rename(columns={'info': 'atom'})
comm = pd.read_csv(f'{networks_dir}/community.csv').rename(columns={'info': 'atom'})

def pivot(measure, src, reduce_regions=False):
    d = src[src['measure'] == measure]
    d = (d.groupby(['subject', 'atom'], as_index=False)['value'].mean()
         if reduce_regions else d[['subject', 'atom', 'value']])
    return d.pivot(index='subject', columns='atom', values='value')[misc.atom_order]

measures = {
    'Efficiency':    pivot('global_efficiency', topo),
    'Complexity':    pivot('von_neumann_entropy', topo),
    'Hubness':       pivot('s_metric', topo),
    'Modularity':    pivot('modularity_group', comm),
    'Participation': pivot('participation_coeff_group', comm, reduce_regions=True),
    'Sim.Yeo':       pivot('ami_vs_yeo', comm),
}

# ranked ordering per measure (to see which atom each claim is about)
for name, piv in measures.items():
    order = piv.mean().sort_values(ascending=False).index
    print(f"% {name}: " + ' > '.join(misc.atoms_names[a] for a in order))

# each atom vs the mean of all other atoms (one-sample t on the per-subject difference)
rows = []
for measure, piv in measures.items():
    for atom in misc.atom_order:
        rest = [a for a in misc.atom_order if a != atom]
        diff = piv[atom].values - piv[rest].mean(axis=1).values
        t, p, cil, ciu, d, bf, dof, power = misc.aug_t_test(diff, np.zeros(len(diff)))
        rows.append({'measure': measure, 'atom': atom,
                     't': t, 'p': p, 'cil': cil, 'ciu': ciu,
                     'd': d, 'bf': float(bf), 'dof': dof, 'power': power})
topo_stats = pd.DataFrame(rows)
topo_stats['p_bh'] = multi.multipletests(topo_stats['p'].values, method='fdr_bh')[1]
topo_stats.to_csv(f'{networks_dir}/topology_atom_vs_rest.csv', index=False)

# LaTeX line for a (measure, atom); direction read from the sign of t (d sign is unreliable)
def cite(measure, atom):
    r = topo_stats[(topo_stats['measure'] == measure) & (topo_stats['atom'] == atom)].iloc[0]
    direction = 'highest' if r['t'] > 0 else 'lowest'
    return (f"{measure} {misc.atoms_texcommands[atom]} {direction}: "
            f"$p_{{\\mathrm{{FDR}}}} = ${misc.write_scientific(r['p_bh'])}")

for measure, atom in [('Complexity', 'Cooperation'), ('Hubness', 'Cooperation'),
                      ('Modularity', 'Cooperation'), ('Sim.Yeo', 'Cooperation'),
                      ('Efficiency', 'Decryption'), ('Modularity', 'Decryption'),
                      ('Sim.Yeo', 'Decryption'), ('Participation', 'Preservation'),
                      ('Complexity', 'Copy'), ('Efficiency', 'Copy')]:
    print(cite(measure, atom))


# %%
# Per-subject Spearman correlation between individual and group PC (across regions)

if not os.path.exists(f'{networks_dir}/pc_indi_group_correlation.csv'):
    pc_ind_nodal = comm_df[comm_df['measure'] == 'participation_coeff_individual'].copy()
    pc_ind_nodal['region'] = pc_ind_nodal['region'].astype(int)
    pc_grp_nodal = comm_df[comm_df['measure'] == 'participation_coeff_group'].copy()
    pc_grp_nodal['region'] = pc_grp_nodal['region'].astype(int)

    subjects = pc_ind_nodal['subject'].unique()

    corr_rows = []
    for sub in subjects:
        for atom_name in misc.atom_order:
            ind_vals = pc_ind_nodal[(pc_ind_nodal['subject'] == sub) & (pc_ind_nodal['atom'] == atom_name)].sort_values('region')['value'].values
            grp_vals = pc_grp_nodal[(pc_grp_nodal['subject'] == sub) & (pc_grp_nodal['atom'] == atom_name)].sort_values('region')['value'].values
            r, _ = spearmanr(ind_vals, grp_vals)
            corr_rows.append({'subject': sub, 'atom': atom_name, 'r': r})

    pc_corr_df = pd.DataFrame(corr_rows)
    pc_corr_df.to_csv(f'{networks_dir}/pc_indi_group_correlation.csv', index=False)
else:
    pc_corr_df = pd.read_csv(f'{networks_dir}/pc_indi_group_correlation.csv')

np.random.seed(42)
fig, ax = plt.subplots(figsize=(5, 3))

for i, atom_name in enumerate(misc.atom_order):
    color = misc.atoms_colors[atom_name]
    vals = pc_corr_df[pc_corr_df['atom'] == atom_name]['r'].values
    ax.scatter(np.full(len(vals), i) + np.random.normal(0, 0.06, len(vals)),
               vals, color=color, alpha=0.6, s=15, zorder=5)
    ax.boxplot(vals, positions=[i], **bp_kw)

ax.axhline(0, color='grey', linestyle='--', linewidth=0.8, zorder=0)
ax.set_xticks(range(len(misc.atom_order)))
ax.set_xticklabels([misc.atoms_names[a] for a in misc.atom_order], rotation=90)
for tick, atom_name in zip(ax.get_xticklabels(), misc.atom_order):
    tick.set_color(misc.atoms_colors[atom_name])
ax.set_ylabel('PC Subject-Group')
ax.set_xlim(-0.5, len(misc.atom_order) - 0.5)
sns.despine(ax=ax)
plt.tight_layout()
plt.show()


# %%
from matplotlib.colors import ListedColormap

for atom in misc.atoms_nr:
    ci = np.load(f'{networks_dir}/group_ci_{atom}.npy')
    n_comm = len(np.unique(ci))
    colors = misc.distinguishable_colors(n_comm, as_hex=False, sort=False,
                          saturation_range=(0, .7), lightness_range=(0, .7))
    cmap = ListedColormap(colors)
    print(f"{atom} - n_comm: {n_comm}", flush=True)
    fig = plot_wholebrain_row(ci.astype(float), atlas=parcellation, brightness=0.4,
                                   cmap=cmap, vmin=0, vmax=n_comm - 1)
    fig.savefig(f'{plots_dir}/communities_{atom}.png', bbox_inches='tight')
    plt.show()


# %%
from matplotlib import cm

pc_grp_nodal = comm_df[comm_df['measure'] == 'participation_coeff_group'].copy()
pc_grp_nodal['region'] = pc_grp_nodal['region'].astype(int)
pc_grp_nodal_mean = pc_grp_nodal.groupby(['atom', 'region'])['value'].mean().reset_index()


cmap = Berlin_20.mpl_colormap
plotcbar = True

for atom in misc.atoms_nr:
    values = pc_grp_nodal_mean[pc_grp_nodal_mean['atom'] == atom].sort_values('region')['value'].values

    vmin = np.nanpercentile(values, 5)
    vmax = np.nanpercentile(values, 95)

    fig = plot_wholebrain_row(values, parcellation, brightness=.4,
                                    cmap=cmap, vmin=vmin, vmax=vmax)
    fig.savefig(f'{plots_dir}/{atom}_pc_group_brainplot.png', bbox_inches='tight')
    plt.close(fig)

    if plotcbar:
        dp = 2 if f'{vmin:.2f}' != f'{vmax:.2f}' else 3
        fig, ax = plt.subplots(1, 1, figsize=(.1, .6))
        cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), cax=ax, orientation='vertical')
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels([f'{vmin:.{dp}f}', f'{vmax:.{dp}f}'], fontsize=13)
        fig.savefig(f'{plots_dir}/{atom}_pc_group_brainplot_cbar.svg',
                    bbox_inches='tight', transparent=True)
        plt.close(fig)


# %%
# mode invariance

from sklearn.metrics.pairwise import cosine_similarity as sk_cos

if not os.path.exists(f'{networks_dir}/cross_atom_cosine_abs.npz'):

    atom_names = misc.atom_order
    n_atoms = len(atom_names)
    off_mask = ~np.eye(n_atoms, dtype=bool)

    def regional_cross_atom_cosine(Ws):
        """Ws: (A, N, N) atom-stacked connectivity for one subject."""
        A, N, _ = Ws.shape
        out = np.full(N, np.nan)
        for r in range(N):
            keep = np.ones(N, dtype=bool); keep[r] = False
            stack = Ws[:, r, keep]
            if np.all(np.linalg.norm(stack, axis=1) > 0):
                out[r] = np.abs(sk_cos(stack))[off_mask].mean()
        return out

    cos_subjects = []
    for perp in misc.subjects:
        try:
            Ws = np.stack([load_connectome(perp, misc.atoms[a]) for a in atom_names])
        except Exception as e:
            print(f'sub-{perp}: load failed - {e}')
            continue
        cos_subjects.append(regional_cross_atom_cosine(Ws))

    cos_arr = np.stack(cos_subjects)
    cos_mean = np.nanmean(cos_arr, axis=0)
    cos_sd = np.nanstd(cos_arr, axis=0)

    np.savez(f'{networks_dir}/cross_atom_cosine_abs.npz',
             cos_mean=cos_mean, cos_sd=cos_sd, cos_subjects=cos_arr,
             atom_names=np.array(atom_names))
else:
    data = np.load(f'{networks_dir}/cross_atom_cosine_abs.npz', allow_pickle=True)
    cos_mean = data['cos_mean']
    cos_sd = data['cos_sd']
    cos_subjects = data['cos_subjects']
    atom_names = data['atom_names']

# print range and mean
print(f"Cross-atom cosine similarity (abs): mean = {cos_mean.mean():.4f}, range = [{cos_mean.min():.4f}, {cos_mean.max():.4f}]")


# %%
# mode invariance brain pot

cmap = misc.hcp_cmap
vmin = np.nanpercentile(cos_mean, 5)
vmax = np.nanpercentile(cos_mean, 95)
fig = plot_full(cos_mean, parcellation, brightness=.4,
                      cmap=cmap, vmin=vmin, vmax=vmax)
fig.savefig(f'{plots_dir}/cross_atom_cosine_abs_mean.png', bbox_inches='tight')

dp = 2 if f'{vmin:.2f}' != f'{vmax:.2f}' else 3
fig, ax = plt.subplots(1, 1, figsize=(.1, .75))
cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), cax=ax, orientation='vertical')
cbar.set_ticks([0, 1])
cbar.set_ticklabels([f'{vmin:.{dp}f}', f'{vmax:.{dp}f}'])
fig.savefig(f'{plots_dir}/cross_atom_cosine_abs_mean_cbar.svg',
            bbox_inches='tight', transparent=True)


# %%
# print top regions for mode invariance
print('-' * 100)
print(f'Mode invariance')
print('-' * 100)
misc.print_topography_extremes(cos_mean, n=50, parc=parcellation, expand=True)


# %%
# per-region cosine for specific atom pairs
pair_contrasts = [
    ('Oneness', 'Cooperation'),
    ('Preservation', 'Migration'),
    ('Copy', 'Unification'),
]

def regional_pair_cosine(W_a, W_b):
    N = W_a.shape[0]
    out = np.full(N, np.nan)
    for r in range(N):
        keep = np.ones(N, dtype=bool); keep[r] = False
        a = W_a[r, keep]; b = W_b[r, keep]
        na = np.linalg.norm(a); nb = np.linalg.norm(b)
        if na > 0 and nb > 0:
            out[r] = np.dot(a, b) / (na * nb)
    return out

pair_cache = f'{networks_dir}/pair_atom_cosine.npz'
if not os.path.exists(pair_cache):
    pair_subjects = {f'{a}__{b}': [] for a, b in pair_contrasts}
    for perp in tqdm(misc.subjects, desc='subjects'):
        try:
            W_by_atom = {}
            atoms_needed = set(a for pair in pair_contrasts for a in pair)
            for a in atoms_needed:
                W_by_atom[a] = load_connectome(perp, misc.atoms[a])
        except Exception as e:
            print(f'sub-{perp}: load failed - {e}')
            continue
        for a, b in pair_contrasts:
            pair_subjects[f'{a}__{b}'].append(regional_pair_cosine(W_by_atom[a], W_by_atom[b]))
    pair_arrays = {k: np.stack(v) for k, v in pair_subjects.items()}
    np.savez(pair_cache, **pair_arrays)
else:
    data = np.load(pair_cache)
    pair_arrays = {k: data[k] for k in data.files}

pair_means = {k: np.nanmean(v, axis=0) for k, v in pair_arrays.items()}
for a, b in pair_contrasts:
    key = f'{a}__{b}'
    vals = pair_means[key]
    print(f'{key}: min={np.nanmin(vals):.3f} max={np.nanmax(vals):.3f} med={np.nanmedian(vals):.3f}')


# %%
# edge-centric architecture: continuous profile across modes (threshold-free)
atoms_used = list(misc.atom_order)
K = len(atoms_used)

def _fill_nans(W):
    return np.where(np.isnan(W), np.nanmin(W), W)

if not os.path.exists(f'{networks_dir}/edge_profile_summary.npz'):

    matrices = np.concatenate(
        [nu.rank_normalise(_fill_nans(group_phiid_matrix[atom]))[np.newaxis]
         for atom in atoms_used],
        axis=0,
    )  # (K, N, N)

    iu = np.triu_indices(n_regions, k=1)
    edge_ranks = matrices[:, iu[0], iu[1]]  # (K, n_edges) in [0,1]
    n_edges = edge_ranks.shape[1]

    mean_rank = edge_ranks.mean(axis=0)
    sd_rank  = edge_ranks.std(axis=0)

    def node_avg(edge_vals):
        s = np.zeros(n_regions); c = np.zeros(n_regions)
        np.add.at(s, iu[0], edge_vals); np.add.at(c, iu[0], 1)
        np.add.at(s, iu[1], edge_vals); np.add.at(c, iu[1], 1)
        return s / np.maximum(c, 1)

    node_mean_rank = node_avg(mean_rank)
    node_sd_rank   = node_avg(sd_rank)

    np.savez(f'{networks_dir}/edge_profile_summary.npz',
             mean_rank=mean_rank, sd_rank=sd_rank,
             node_mean_rank=node_mean_rank, node_sd_rank=node_sd_rank,
             atoms=np.array(atoms_used))
    
else:
    data = np.load(f'{networks_dir}/edge_profile_summary.npz', allow_pickle=True)
    mean_rank = data['mean_rank']
    sd_rank = data['sd_rank']
    node_mean_rank = data['node_mean_rank']
    node_sd_rank = data['node_sd_rank']
    atoms_used = data['atoms']


# %%
# 2D distribution of edges
cmap = cmr.amber
fig, ax = plt.subplots(figsize=(2.8, 2.5))
sns.histplot(x=mean_rank, y=sd_rank, bins=1000, cmap=cmap, vmax=15, cbar=False, ax=ax)
ax.set_xlabel(r'Consensus ($\mu$)')
ax.set_ylabel(r'Contrast ($\sigma$)')
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
sns.despine()
plt.tight_layout()
fig.savefig(f'{plots_dir}/edge_mean_sd_hist.png', bbox_inches='tight')
fig.savefig(f'{plots_dir}/edge_mean_sd_hist.svg', bbox_inches='tight', transparent=True)


# %%
# top-1000 mean-rank and top-1000 sigma-rank edges as side-view brain networks
# uses mean_rank / sd_rank from cell 19 (edge_profile_summary)

group_phiid_path = f'{main_dir}/{parcellation}/group_phiid_matrix.npy'
group_phiid_matrix = np.load(group_phiid_path, allow_pickle=True).item()

kwargs_multilayer = dict(node_color="#DDDADA", edge_color="#2A2A2A", node_edgecolor="#3A3939", node_linewidth=.5,
              edge_alpha=(0, 1),  edge_width=(0.5, 5.0), node_size=(25, 200))

iu = np.triu_indices(n_regions, k=1)
coords_arr = atlas_info[['x', 'y', 'z']].values[:n_regions, :]

def _edges_to_matrix(values, top_n=1000):
    top = np.argsort(values)[::-1][:top_n]
    W = np.zeros((n_regions, n_regions), dtype=float)
    W[iu[0][top], iu[1][top]] = values[top]
    W[iu[1][top], iu[0][top]] = values[top]
    return W

panels = [
    ('mean', mean_rank, 'edge_top1000_mean'),
    ('sd',   sd_rank,  'edge_top1000_sd'),
]

for label, vals, fname in panels:
    W = _edges_to_matrix(vals, top_n=1000)
    fig, ax = plot_brain_network(W, coords=coords_arr,
                                 edge_scores=W, node_scores=W.sum(axis=0),
                                 view=['left','right'], **kwargs_multilayer)
    fig.savefig(f'{plots_dir}/{fname}.png', bbox_inches='tight', transparent=True)
    plt.show()


# %%
# top-500 edges per atom, overlaid with atom-specific colors
top_n_per_atom = 500

# rebuild per-atom edge rank stack (same as cell 18 logic)
iu = np.triu_indices(n_regions, k=1)
atom_mats = {
    a: nu.rank_normalise(np.where(np.isnan(group_phiid_matrix[a]),
                                   np.nanmin(group_phiid_matrix[a]),
                                   group_phiid_matrix[a]))
    for a in atoms_used
}
atom_top_mats = {}
for a in atoms_used:
    ranks = atom_mats[a][iu[0], iu[1]]
    top = np.argsort(ranks)[::-1][:top_n_per_atom]
    W = np.zeros((n_regions, n_regions), dtype=float)
    W[iu[0][top], iu[1][top]] = ranks[top]
    W[iu[1][top], iu[0][top]] = ranks[top]
    atom_top_mats[a] = W

# combined node size = total strength across all top-100 edges
combined_node_strength = np.sum(
    [W.sum(axis=0) for W in atom_top_mats.values()], axis=0
)

views = ['left', 'right']
fig, axes = plt.subplots(1, len(views), figsize=(8, 4),
                         gridspec_kw={'wspace': 0})
axes = list(np.atleast_1d(axes))


kwargs_overlay = dict(node_color="#DDDADA", edge_color="#2A2A2A", node_edgecolor="#3A3939", node_linewidth=.5,
              edge_alpha=(0.3, 0.9),  edge_width=(0.4, 2.0), node_size=(5, 150))


for a in atoms_used[::-1]:  # plot in reverse order so that first atoms are on top
    color = misc.atoms_colors[a]
    kwargs_overlay['edge_color'] = color
    plot_brain_network(
        atom_top_mats[a],
        coords=coords_arr,
        edge_scores=atom_top_mats[a],
        node_scores=combined_node_strength,
        view=views,
        ax=axes,
        **kwargs_overlay,
    )

fig.savefig(f'{plots_dir}/edge_top_{top_n_per_atom}_per_atom.png', bbox_inches='tight', transparent=True)
plt.show()


# %%
tables_dir = f'{plots_dir}/tables'
os.makedirs(tables_dir, exist_ok=True)

topo_src = pd.read_csv(f'{networks_dir}/topology_global.csv').rename(columns={'info': 'atom'})
comm_src = pd.read_csv(f'{networks_dir}/community.csv').rename(columns={'info': 'atom'})

def _pivot(measure, src, reduce_regions=False):
    d = src[src['measure'] == measure]
    d = (d.groupby(['subject', 'atom'], as_index=False)['value'].mean()
         if reduce_regions else d[['subject', 'atom', 'value']])
    return d.pivot(index='subject', columns='atom', values='value')[misc.atom_order]

topo_measures = {
    'Efficiency': _pivot('global_efficiency', topo_src),
    'Complexity': _pivot('von_neumann_entropy', topo_src),
    'Hubness': _pivot('s_metric', topo_src),
    'Modularity': _pivot('modularity_group', comm_src),
    'Participation': _pivot('participation_coeff_group', comm_src, reduce_regions=True),
    'Sim.Yeo': _pivot('ami_vs_yeo', comm_src),
}
measure_label = {'Sim.Yeo': 'Similarity to Yeo-17'}

topo_stats = pd.read_csv(f'{networks_dir}/topology_atom_vs_rest.csv')

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

def _fmt_mean(vals):
    # s-metric runs to 1e11, so anything outside a readable range goes to mantissa
    # and exponent rather than being silently rounded to something meaningless
    m, s = float(np.mean(vals)), float(np.std(vals, ddof=1))
    if m != 0 and (abs(m) >= 1e4 or abs(m) < 1e-3):
        e = int(np.floor(np.log10(abs(m))))
        return r'$(%.2f \pm %.2f)\times10^{%d}$' % (m / 10 ** e, s / 10 ** e, e)
    return r'$%.3f \pm %.3f$' % (m, s)

rows = []
for i, (measure, piv) in enumerate(topo_measures.items()):
    strut = '' if i == 0 else r'\rule{0pt}{3.2ex}'
    label = measure_label.get(measure, measure)
    rows.append(r'\multicolumn{6}{l}{%s\textit{%s}} \\' % (strut, label))
    sub = topo_stats[topo_stats['measure'] == measure].set_index('atom')
    for atom in misc.atom_order:
        r = sub.loc[atom]
        direction = r'$\uparrow$' if r['t'] > 0 else r'$\downarrow$'
        rows.append(r'\quad %s & %s & %s %s & %.2f & %s & %s \\' % (
            misc.atoms_texcommands[atom],
            _fmt_mean(piv[atom].values),
            direction,
            _sci(abs(float(r['t']))),
            abs(float(r['d'])),
            _sci(r['bf']),
            _sci(r['p_bh']),
        ))

tex = r"""%% generated by networks_s.ipynb, do not edit by hand
\begin{table*}[!ht]
\centering
%% NO p{} COLUMNS. revtex4-1 patches the array package internals, and against
%% array >= 2024 (TeX Live 2025) the patch no longer applies -- the class says
%% "Unrecognized array package ... proceeding with fingers crossed" and any p{}
%% column then throws an endless "Extra \or." from \@testpach. l/c columns take
%% an earlier branch and are safe
%% thinner rules than the 0.4pt default, and \small confined to the tabular so
%% the caption keeps the normal body size
\begingroup
\setlength{\arrayrulewidth}{0.2pt}
\small
\begin{tabular}{lccccc}
\hline
Mode & Value (mean $\pm$ SD) & $t$ & $|d|$ & $BF_{10}$ & $p_{\mathrm{FDR}}$ \\
\hline
%(rows)s
\hline
\end{tabular}
\endgroup
%% the caption renders wherever \caption is issued, so putting it after the
%% tabular places it below the table
\caption{\textbf{Global topology of networks of information dynamics.} Each mode
tested against the mean of the nine others, by a one-sample $t$-test on the per-participant
difference ($n=%(n)d$, $\mathrm{df}=%(dof)d$). \textit{Value} is the raw measure, averaged over
participants. The arrow on $t$ gives the direction, $\uparrow$ where the mode exceeds the
mean of the others and $\downarrow$ where it falls below.
$p$-values are FDR-corrected by the Benjamini-Hochberg procedure across all %(ncell)d cells.
Bayes factors are JZS Bayes factors under a Cauchy prior of scale $r=0.707$.}
\label{tab:topology}
\end{table*}
""" % {'rows': '\n'.join(rows), 'n': len(topo_measures['Efficiency']),
       'dof': int(topo_stats['dof'].iloc[0]), 'ncell': len(topo_stats)}

out = f'{tables_dir}/topology_table.tex'
with open(out, 'w') as fh:
    fh.write(tex)
print(f'wrote {out}: {len(topo_stats)} cells over {len(topo_measures)} measures')
print(f"significant at p_FDR < 0.05: {int((topo_stats['p_bh'] < 0.05).sum())}/{len(topo_stats)}")
