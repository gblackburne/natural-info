# %%
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
import statsmodels.stats.multitest as multi
import warnings
import itertools
import importlib
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

import misc as misc
importlib.reload(misc)

subjects = misc.subjects
parcellation = misc.parcellation
n_regions = misc.n_regions
regions = misc.regions

MRI_path = '/home/george/Research/fMRI_general'
atlas_info = pd.read_csv(f'{MRI_path}/atlases/schaefer_extend_atlases/{parcellation}.csv')

main_dir = f'results/main'
if not os.path.exists(main_dir):
    os.makedirs(main_dir)

plots_dir = f'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

import sys
sys.path.append(f'{MRI_path}')
from cortexcanvas import (plot_wholebrain, plot_full, 
                          plot_wholebrain_row, plot_wholebrain_row_2,
                          plot_grand_ensemble)

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
for perp in subjects:
    mat = load_connectome(perp, atom_indices=[0], fix_nans=False)
    off_diag = mat[np.triu_indices_from(mat, k=1)]
    if np.isnan(off_diag).any():
        print(f'sub-{perp} has nans in the connectome')


# %%
# since summing atoms for uniques is somewhat trivial they are higher
# take mean instead of sum?

# repeated-measures test for differences in atom values across subjects
rows = []
for atom in misc.atom_order:
    atom_indices = misc.atoms[atom]
    for perp in subjects:
        mat = load_connectome(perp, atom_indices, fix_nans=False)
        if mat is None:
            continue
        tri_idx = np.triu_indices(mat.shape[0], k=1)
        rows.append({'subject': perp, 'atom': atom,
                     'value': np.nanmean(mat[tri_idx])})

df_long = pd.DataFrame(rows)

# omnibus repeated-measures ANOVA (Greenhouse-Geisser if sphericity is violated)
rm = pg.rm_anova(data=df_long, dv='value', within='atom',
                 subject='subject', correction=True)
# post-hoc paired comparisons,
post = pg.pairwise_tests(data=df_long, dv='value', within='atom',
                         subject='subject', padjust='fdr_bh')

r = rm.iloc[0]
pcol = 'p_GG_corr'
min_p = post['p_corr'].min()
max_p = post['p_corr'].max()

print(f"(rm-ANOVA: {misc.write_p(r[pcol])}, $\\eta^2_G$ = {r['ng2']:.3f}; "
      f"pairwise comparisons: {misc.write_p(min_p, sub='FDR')} "
      f"to {misc.write_p(max_p, sub='FDR')})")


# %%
# collect all edge values across all subjects for each atom, compute & save histograms
hist_save_path = f'{main_dir}/{parcellation}/atom_pdfs.npy'

if not os.path.exists(hist_save_path):
    all_values = {}
    for atom, atom_indices in misc.atoms.items():
        vals = []
        for perp in subjects:
            try:
                mat = load_connectome(perp, atom_indices, fix_nans=False)
                tri_idx = np.triu_indices(mat.shape[0], k=1)
                vals.append(mat[tri_idx])
            except Exception:
                continue
        all_values[atom] = np.concatenate(vals)

    global_min = min(np.nanmin(v) for v in all_values.values())
    global_max = max(np.nanmax(v) for v in all_values.values())
    bins = np.linspace(global_min, global_max, 1001)

    atom_pdfs = {}
    for atom, values in all_values.items():
        counts, edges = np.histogram(values[~np.isnan(values)], bins=bins)
        counts = counts / counts.sum()
        centers = (edges[:-1] + edges[1:]) / 2
        atom_pdfs[atom] = {'counts': counts, 'edges': edges, 'centers': centers}

    np.save(hist_save_path, atom_pdfs)
else:
    atom_pdfs = np.load(hist_save_path, allow_pickle=True).item()

fig, ax = plt.subplots(figsize=(8, 4))
for atom in misc.atom_order:
    d = atom_pdfs[atom]
    ax.plot(d['centers'], d['counts'], label=misc.atoms_names[atom],
            color=misc.atoms_colors[atom], linewidth=1.5)

ax.set_xlabel('Bits')
ax.set_ylabel('PDF')
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(3))
plt.gca().yaxis.set_major_locator(plt.MaxNLocator(3))
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, frameon=False)
sns.despine()
plt.tight_layout()
plt.show()

# radical radial brev
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image as PILImage

ap_o = [
    'Cooperation', 
    'Preservation', 
    'Copy', 
    'Encryption',
    'Unification',
    'Oneness',
    'Migration',
    'Erasure',
    'Decryption',
    'Dissolution'
]

# radial plot parameters
violin_width  = 1.2
label_fontsize = 25
icon_size     = 110

n_atoms = len(ap_o)
theta = np.linspace(0, 2 * np.pi, n_atoms, endpoint=False)
width = (2 * np.pi) / n_atoms * violin_width

def radial_rotation(angle_rad):
    """Rotate to face radially outward, clamped to ±90° so icons stay readable."""
    deg = -np.degrees(angle_rad)
    deg = (deg + 180) % 360 - 180
    if deg < -90:
        deg += 180
    elif deg > 90:
        deg -= 180
    return deg

# load and downsample icons — source PNGs are ~6000px; cap at 2400px so icons
# stay sharp at 1600 DPI while keeping memory use reasonable (~90MB each)
icon_imgs = {}
for atom in ap_o:
    atom_code = misc.group_atoms[atom][0]
    img_arr = plt.imread(f'plots/atom_rep/{atom_code}.png')
    pil_img = PILImage.fromarray((img_arr * 255).astype(np.uint8)).convert('RGBA')
    max_dim = max(pil_img.size)
    if max_dim > 2400:
        scale = 2400 / max_dim
        pil_img = pil_img.resize(
            (int(pil_img.width * scale), int(pil_img.height * scale)),
            PILImage.LANCZOS
        )
    icon_imgs[atom] = pil_img

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='polar')
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

for i, (atom, angle) in enumerate(zip(ap_o, theta)):
    d = atom_pdfs[atom]
    color = misc.atoms_colors[atom]
    r = d['centers']
    w = (d['counts'] / d['counts'].max()) * (width / 2)
    ax.fill_betweenx(r, angle - w, angle + w, color=color)

rmax = max(d['centers'].max() for d in atom_pdfs.values())
ax.set_ylim(top=rmax * 1.75)

ax.set_xticks(theta)
ax.set_xticklabels([])
ax.set_yticks([-1, 0, 1])
ax.set_yticklabels([])          # hide auto labels; drawn manually below
ax.tick_params(axis='y', length=0)
ax.spines['polar'].set_visible(False)
ax.grid(False)

pres_theta = theta[ap_o.index('Preservation')]
ax.plot([pres_theta, pres_theta], [ax.get_ylim()[0], rmax],
        color='black', linewidth=0.8, zorder=0, alpha=0.35)

tick_hw = 0.025
for r_tick, label in zip([-1, 0, 1], ['-1', '0', '1']):
    ax.plot([pres_theta - tick_hw, pres_theta + tick_hw], [r_tick, r_tick],
            color='black', linewidth=1.2, zorder=6)
    ax.annotate(label, xy=(pres_theta, r_tick),
                xytext=(6, -4), textcoords='offset points',
                ha='left', va='center', fontsize=label_fontsize, zorder=7)

icon_r_frac = 0.4
text_r_frac  = icon_r_frac + 0.135

for atom, angle in zip(ap_o, theta):
    cart_angle = np.pi / 2 - angle
    x_ax = 0.5 + icon_r_frac * np.cos(cart_angle)
    y_ax = 0.5 + icon_r_frac * np.sin(cart_angle)

    rot_deg = radial_rotation(angle)
    # zoom from pre-rotation size so all icons render at the same content scale
    # (rotation with expand=True grows the bounding box, which would otherwise
    # shrink unrotated icons like Cooperation/Oneness relative to the rest)
    zoom = icon_size / max(icon_imgs[atom].size)
    rotated = icon_imgs[atom].rotate(rot_deg, expand=True, fillcolor=(0, 0, 0, 0))
    imagebox = OffsetImage(np.array(rotated).astype(np.float32) / 255.0, zoom=zoom)
    ab = AnnotationBbox(imagebox, (x_ax, y_ax), xycoords='axes fraction',
                        frameon=False, pad=0)
    ab.set_clip_on(False)
    ax.add_artist(ab)

    tx = 0.5 + text_r_frac * np.cos(cart_angle)
    ty = 0.5 + text_r_frac * np.sin(cart_angle)
    ax.text(tx, ty, misc.atoms_names[atom], transform=ax.transAxes,
            ha='center', va='center', fontsize=label_fontsize, rotation=rot_deg,
            rotation_mode='anchor')

plt.savefig(f'{plots_dir}/atom_radial_pdf.png', dpi=1600)
plt.close()


# %%
# if group-level matrices not computed then compute them

def load_fc(perp, fix_nans=True):
    out = np.load(f'{main_dir}/{parcellation}/sub-{perp}/pearson_matrix.npy')
    if fix_nans:
        if np.isnan(out).any():
            min_finite = np.nanmin(out)
            out = np.where(np.isnan(out), min_finite, out)
    return out

group_phiid_path = f'{main_dir}/{parcellation}/group_phiid_matrix.npy'
if not os.path.exists(group_phiid_path):
    group_phiid_matrix = {}
    for atom_name, atom_indices in misc.atoms.items():
        matrices = [m for m in (load_connectome(perp, atom_indices, fix_nans=False) for perp in subjects) if m is not None]
        group_phiid_matrix[atom_name] = np.nanmean(np.stack(matrices), axis=0)
    np.save(group_phiid_path, group_phiid_matrix)
else:
    group_phiid_matrix = np.load(group_phiid_path, allow_pickle=True).item()

group_phiid_path = f'{main_dir}/{parcellation}/group_phiid_regions.npy'
if not os.path.exists(group_phiid_path):
    group_phiid_region = {}
    for atom_name, atom_indices in misc.atoms.items():
        matrices = [m for m in (np.nanmean(load_connectome(perp, atom_indices, fix_nans=False), axis=0) for perp in subjects) if m is not None]
        group_phiid_region[atom_name] = np.nanmean(np.stack(matrices), axis=0)
    np.save(group_phiid_path, group_phiid_region)
else:
    group_phiid_region = np.load(group_phiid_path, allow_pickle=True).item()


# %%
gl_corrs = pd.DataFrame(index=misc.atom_order, columns=misc.atom_order)
for pair in itertools.combinations(misc.atom_order, 2):
    atom1, atom2 = pair
    corr = np.corrcoef(np.triu(group_phiid_matrix[atom1], k=1).flatten(), np.triu(group_phiid_matrix[atom2], k=1).flatten())[0, 1]
    gl_corrs.loc[atom1, atom2] = corr
    gl_corrs.loc[atom2, atom1] = corr
gl_corrs = gl_corrs.astype(float)
gl_corrs = gl_corrs.mask(np.eye(len(gl_corrs), dtype=bool), 1)

cmap2 = Berlin_20.mpl_colormap

fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(gl_corrs.astype(float), cmap=Vik_3.mpl_colormap, 
               vmin=-1, vmax=1, aspect='equal')
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

if not os.path.exists(f'{main_dir}/{parcellation}/swise_matrix_corrs.npy'):
    subject_corrs = np.nan*np.zeros((len(misc.atom_order), len(misc.atom_order), len(subjects)))
    for subject in subjects:
        for pair in itertools.combinations(misc.atom_order, 2):
            atom1, atom2 = pair
            mat1 = load_connectome(subject, misc.atoms[atom1], fix_nans=False)
            mat2 = load_connectome(subject, misc.atoms[atom2], fix_nans=False)
            if mat1 is not None and mat2 is not None:
                nan_mask = np.isnan(mat1) | np.isnan(mat2)
                corr = np.corrcoef(np.triu(mat1, k=1)[~nan_mask].flatten(), np.triu(mat2, k=1)[~nan_mask].flatten())[0, 1]
                idx1, idx2 = misc.atom_order.index(atom1), misc.atom_order.index(atom2)
                subject_corrs[idx1, idx2, subject-1] = corr
                subject_corrs[idx2, idx1, subject-1] = corr
    np.save(f'{main_dir}/{parcellation}/swise_matrix_corrs.npy', subject_corrs)
else:
    subject_corrs = np.load(f'{main_dir}/{parcellation}/swise_matrix_corrs.npy')

fig, ax = plt.subplots(figsize=(5, 5))
mean_corrs = np.nanmean(subject_corrs, axis=2)
np.fill_diagonal(mean_corrs, 1)
im = ax.imshow(mean_corrs, cmap=Vik_3.mpl_colormap, 
               vmin=-1, vmax=1, aspect='equal')
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


# %%
# plot topographies

plotcbar = True

for atom in misc.atoms:
    values = group_phiid_region[atom]

    vmin = np.nanpercentile(values, 5); vmax = np.nanpercentile(values, 95)
    if atom in ['Cooperation', 'Oneness']:
        fig = plot_wholebrain(values, parcellation, brightness=.4,
                              cmap=misc.con_cmap, vmin=vmin, vmax=vmax)
    else:
        fig = plot_wholebrain_row_2(values, parcellation, brightness=.4,
                                    cmap=misc.con_cmap, vmin=vmin, vmax=vmax)
    fig.savefig(f'{plots_dir}/{atom}_global_brainplot.png', bbox_inches='tight')
    plt.close(fig)

    if plotcbar:
        dp = 2 if f'{vmin:.2f}' != f'{vmax:.2f}' else 3
        if atom in ['Cooperation', 'Oneness']:
             fig, ax = plt.subplots(1, 1, figsize=(.75, .1))
             cbar = plt.colorbar(cm.ScalarMappable(cmap=misc.con_cmap), cax=ax, orientation='horizontal')
             cbar.set_ticks([0, 1])
             cbar.set_ticklabels([f'{vmin:.{dp}f}', f'{vmax:.{dp}f}'])
             fig.savefig(f'{plots_dir}/{atom}_global_brainplot_cbar_horizontal.svg', bbox_inches='tight',
                         transparent=True)
             plt.close(fig)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(.1, .75))
            cbar = plt.colorbar(cm.ScalarMappable(cmap=misc.con_cmap), cax=ax, orientation='vertical')
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels([f'{vmin:.{dp}f}', f'{vmax:.{dp}f}'])
            fig.savefig(f'{plots_dir}/{atom}_global_brainplot_cbar.svg', bbox_inches='tight',
                        transparent=True)
            plt.close(fig)


# %%
# write top perfoming regions
for atom in misc.atoms:
    print('-' * 100)
    print(f'{atom}:')
    print('-' * 100)
    values = group_phiid_region[atom]
    misc.print_topography_extremes(values, n=25, parc=parcellation, expand=True)
    print('\n' * 4 + '-' * 100 + '\n' * 2)


# %%
# representational complexity
#
# Adapted from Ehrlich et al. (2023, TMLR): C = (1/I) * sum_k(pi_k * m_k)
# m_k = "degree of synergy" - minimum number of sources needed to access atom k.
# Three variants:
#   sources : m based on past (source) side - r*/x*/y* -> 1,  s*  -> 2 (original for PID)
#   targets : m based on future (target) side - *r/*a/*b -> 1, *s  -> 2
#   global  : m = (m_sources + m_targets) / 2

m_k_sources = np.array([1, 1, 1, 1,   # r* (rtr rta rtb rts)
                         1, 1, 1, 1,   # x* (xtr xta xtb xts)
                         1, 1, 1, 1,   # y* (ytr yta ytb yts)
                         2, 2, 2, 2],  # s* (str sta stb sts)
                        dtype=np.float64)

m_k_targets = np.array([1, 1, 1, 2,   # r* row: *r *a *b *s
                         1, 1, 1, 2,   # x* row: *r *a *b *s
                         1, 1, 1, 2,   # y* row: *r *a *b *s
                         1, 1, 1, 2],  # s* row: *r *a *b *s
                        dtype=np.float64)

m_k_global = (m_k_sources + m_k_targets) / 2

assert all((misc.phiid_atoms[k][0] == 's') == (m_k_sources[k] == 2) for k in range(16)), \
    "m_k_sources does not match phiid_keys source-side types"
assert all((misc.phiid_atoms[k][2] == 's') == (m_k_targets[k] == 2) for k in range(16)), \
    "m_k_targets does not match phiid_keys target-side types"

def compute_repr_complexity(perp, m_k):
    mat = np.load(f'{main_dir}/{parcellation}/sub-{perp}/phiid_matrix.npy').astype(np.float64)
    total    = np.nansum(mat, axis=2)
    weighted = (mat * m_k[np.newaxis, np.newaxis, :]).sum(axis=2)
    with np.errstate(invalid='ignore', divide='ignore'):
        C = np.where(total != 0, weighted / total, np.nan)
    return C

_rc = {}
for variant, m_k in [('sources', m_k_sources), ('targets', m_k_targets), ('global', m_k_global)]:
    mpath = f'{main_dir}/{parcellation}/group_repr_complexity_{variant}_matrix.npy'
    rpath = f'{main_dir}/{parcellation}/group_repr_complexity_{variant}_regions.npy'

    if not os.path.exists(mpath):
        matrices = []
        for perp in subjects:
            try:
                matrices.append(compute_repr_complexity(perp, m_k))
            except Exception:
                continue
        mat_out = np.nanmean(np.stack(matrices), axis=0)
        np.save(mpath, mat_out)
    else:
        mat_out = np.load(mpath)

    if not os.path.exists(rpath):
        regional_means = []
        for perp in subjects:
            try:
                C = compute_repr_complexity(perp, m_k)
                regional_means.append(np.nanmean(C, axis=0))
            except Exception:
                continue
        reg_out = np.nanmean(np.stack(regional_means), axis=0)
        np.save(rpath, reg_out)
    else:
        reg_out = np.load(rpath)

    _rc[variant] = {'matrix': mat_out, 'region': reg_out}
    print(f'Complexity ({variant}) matrix: [{np.nanmin(mat_out):.3f}, {np.nanmax(mat_out):.3f}]  '
          f'region: [{np.nanmin(reg_out):.3f}, {np.nanmax(reg_out):.3f}]')

group_repr_complexity_sources_matrix = _rc['sources']['matrix']
group_repr_complexity_sources_region = _rc['sources']['region']
group_repr_complexity_targets_matrix = _rc['targets']['matrix']
group_repr_complexity_targets_region = _rc['targets']['region']
group_repr_complexity_global_matrix  = _rc['global']['matrix']
group_repr_complexity_global_region  = _rc['global']['region']
del _rc

# computational diversity
#
# i.e atomic participation ratio or Renyi-2 diversity.
# PR = (1 / sum_k p_k^2) / n,  where p_k = a_k^2 / sum_j a_j^2
# Range: 1/n (all mass on one atom) to 1 (uniform across all atoms).

def compute_comp_diversity(perp):
    mat = np.load(f'{main_dir}/{parcellation}/sub-{perp}/phiid_matrix.npy').astype(np.float64)
    w = mat ** 2
    s = w.sum(axis=2)
    with np.errstate(invalid='ignore', divide='ignore'):
        p = np.where(s[..., np.newaxis] != 0, w / s[..., np.newaxis], 1.0 / 16)
        D = np.where(s != 0, 1.0 / (np.sum(p ** 2, axis=2) * 16), 1.0)
    return D

cd_matrix_path = f'{main_dir}/{parcellation}/group_comp_diversity_matrix.npy'
if not os.path.exists(cd_matrix_path):
    matrices = []
    for perp in subjects:
        try:
            matrices.append(compute_comp_diversity(perp))
        except Exception:
            continue
    group_comp_diversity_matrix = np.nanmean(np.stack(matrices), axis=0)
    np.save(cd_matrix_path, group_comp_diversity_matrix)
else:
    group_comp_diversity_matrix = np.load(cd_matrix_path)

cd_region_path = f'{main_dir}/{parcellation}/group_comp_diversity_regions.npy'
if not os.path.exists(cd_region_path):
    regional_means = []
    for perp in subjects:
        try:
            D = compute_comp_diversity(perp)
            regional_means.append(np.nanmean(D, axis=0))
        except Exception:
            continue
    group_comp_diversity_region = np.nanmean(np.stack(regional_means), axis=0)
    np.save(cd_region_path, group_comp_diversity_region)
else:
    group_comp_diversity_region = np.load(cd_region_path)

print(f'Diversity matrix: [{np.nanmin(group_comp_diversity_matrix):.3f}, {np.nanmax(group_comp_diversity_matrix):.3f}]  '
      f'region: [{np.nanmin(group_comp_diversity_region):.3f}, {np.nanmax(group_comp_diversity_region):.3f}]')


# %%
# plot representational complexity and diversity

for variant in ['sources', 'targets', 'global']:
    values = globals()[f'group_repr_complexity_{variant}_region']
    vmin = np.nanpercentile(values, 1)
    vmax = np.nanpercentile(values, 99)
    fig = plot_full(values, parcellation, brightness=.4,
                    cmap=misc.hcp_cmap, vmin=vmin, vmax=vmax)
    fig.savefig(f'{plots_dir}/rcomplex_{variant}_brainplot.png', bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=(.1, .75))
    cbar = plt.colorbar(cm.ScalarMappable(cmap=misc.hcp_cmap), cax=ax, orientation='vertical')
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels([f'{vmin:.2f}', f'{vmax:.2f}'])
    fig.savefig(f'{plots_dir}/rcomplex_{variant}_cbar.svg', bbox_inches='tight',
                transparent=True)
    plt.close()

values = group_comp_diversity_region
vmin = np.nanpercentile(values, 1)
vmax = np.nanpercentile(values, 99)
fig = plot_full(values, parcellation, brightness=.4,
                cmap=misc.hcp_cmap, vmin=vmin, vmax=vmax)
fig.savefig(f'{plots_dir}/cdiversity_brainplot.png', bbox_inches='tight')

fig, ax = plt.subplots(1, 1, figsize=(.1, .75))
cbar = plt.colorbar(cm.ScalarMappable(cmap=misc.hcp_cmap), cax=ax, orientation='vertical')
cbar.set_ticks([0, 1])
cbar.set_ticklabels([f'{vmin:.2f}', f'{vmax:.2f}'])
fig.savefig(f'{plots_dir}/cdiversity_cbar.svg', bbox_inches='tight',
            transparent=True)
plt.close()


# %%
# print top regions for diversity

print('-' * 100)
print(f'Computational diversity:')
print('-' * 100)
misc.print_topography_extremes(group_comp_diversity_region, n=25, parc=parcellation, expand=True)


# %%
def get_synred_grad(regional_syn, regional_red):
    syn_rank = np.argsort(np.argsort(regional_syn))+1
    red_rank = np.argsort(np.argsort(regional_red))+1
    grad = syn_rank - red_rank
    return grad

group_syn = group_phiid_region['Cooperation']
group_red = group_phiid_region['Oneness']
syn_red_grad = get_synred_grad(group_syn, group_red)
np.save(f'{main_dir}/{parcellation}/group_syn_red_grad.npy', syn_red_grad)

print('Min:', np.nanmin(syn_red_grad))
print('Max:', np.nanmax(syn_red_grad))


# %%
fig = plot_grand_ensemble(syn_red_grad,
                         atlas='extended_schaefer_1000',
                         cmap=misc.sr_full_cmap,
                         brightness=.4,
                         vmin=-250, vmax=250)

plt.savefig(f'{plots_dir}/SynRed_Grad_wholebrain.png', bbox_inches='tight')


# %%
# print top regions for syn-red gradient
print('-' * 100)
print(f'Synergy-Redundancy Gradient:')
print('-' * 100)
misc.print_topography_extremes(syn_red_grad, n=50, parc=parcellation, expand=True)


# %%
# plot colourbar horizontally with ticks
fig, ax = plt.subplots(1, 1, figsize=(3, .2))
from matplotlib import cm
cbar = plt.colorbar(cm.ScalarMappable(cmap=misc.sr_full_cmap), cax=ax, orientation='horizontal')
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels([r'$R \rightarrow R$', '', r'$S \rightarrow S$'])
cbar.ax.tick_params(labelsize=15)
ax.set_title('Synergy-Redundancy Rank', fontsize=15) 
plt.savefig(f'{plots_dir}/SynRed_Grad_cbar.png', bbox_inches='tight', transparent=True)

fig, ax = plt.subplots(1, 1, figsize=(2.25, .2))
cbar = plt.colorbar(
    cm.ScalarMappable(cmap=misc.sr_full_cmap), 
    cax=ax, 
    orientation='horizontal'
)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels([r'$\leq$-250', '', r'$\geq$250'])
cbar.ax.tick_params(labelsize=15)
# label
cbar.set_label('Rank', fontsize=15, labelpad=-10)
secax = cbar.ax.secondary_xaxis('top')
secax.set_xticks([0, 1])
secax.set_xticklabels([r'$R \rightarrow R$', r'$S \rightarrow S$'])
secax.tick_params(labelsize=15)
plt.savefig(f'{plots_dir}/SynRed_Grad_cbar_v2.png', bbox_inches='tight', transparent=True)
plt.show()


# %%
# Individual-level synergy-redundancy rank, stratified by
# Yeo network, Mesulam type, and Wang/VTPM retinotopic area
import wang_utils as wu

network_dict = {
    'ContA': 'Cont-a', 'ContB': 'Cont-b', 'ContC': 'Cont-c',
    'DefaultA': 'DMN-a', 'DefaultB': 'DMN-b', 'DefaultC': 'DMN-c',
    'DorsAttnA': 'DAN-a', 'DorsAttnB': 'DAN-b',
    'TempPar': 'TPN',
    'LimbicA': 'Lim-a', 'LimbicB': 'Lim-b',
    'SalVentAttnA': 'SalVAN-a', 'SalVentAttnB': 'SalVAN-b',
    'SomMotA': 'SMN-a', 'SomMotB': 'SMN-b',
    'VisCent': 'VisCent', 'VisPeri': 'VisPeri',
    'Subcort': 'Subcortex'
}

network_array = atlas_info['network'].values
networks = np.unique(network_array)
type_array = np.load(f'{MRI_path}/microstructure/mesulam_{parcellation}.npy', allow_pickle=True)
types = np.unique(type_array[~pd.isnull(type_array)])

yeo_csv_path     = f'{main_dir}/{parcellation}/indiv_rank_yeo_networks.csv'
mesulam_csv_path = f'{main_dir}/{parcellation}/indiv_rank_mesulam.csv'
individual_rank_path = f'{main_dir}/{parcellation}/indiv_syn_red_grad.npy'
indiv_syn_red_grad = np.full((n_regions, len(subjects)), np.nan)

if not (os.path.exists(yeo_csv_path) and os.path.exists(mesulam_csv_path)):
    yeo_rows, mesulam_rows = [], []
    for perp in subjects:
        try:
            syn_mat = load_connectome(perp, misc.atoms['Cooperation'], fix_nans=False)
            red_mat = load_connectome(perp, misc.atoms['Oneness'],     fix_nans=False)
        except Exception:
            continue
        syn_regional = np.nanmean(syn_mat, axis=0)
        red_regional = np.nanmean(red_mat, axis=0)
        grad = get_synred_grad(syn_regional, red_regional)
        indiv_syn_red_grad[:, perp-1] = grad
        for network in networks:
            idx = np.where(network_array == network)[0]
            yeo_rows.append({'subject': perp, 'network': network_dict[network],
                             'mean_rank': np.nanmean(grad[idx])})
        for type_ in types:
            idx = np.where(type_array == type_)[0]
            mesulam_rows.append({'subject': perp, 'type': type_,
                                 'mean_rank': np.nanmean(grad[idx])})
    rank_yeo_df = pd.DataFrame(yeo_rows)
    rank_mesulam_df = pd.DataFrame(mesulam_rows)
    rank_yeo_df.to_csv(yeo_csv_path, index=False)
    rank_mesulam_df.to_csv(mesulam_csv_path, index=False)
    np.save(individual_rank_path, indiv_syn_red_grad)
else:
    rank_yeo_df = pd.read_csv(yeo_csv_path)
    rank_mesulam_df = pd.read_csv(mesulam_csv_path)
    indiv_syn_red_grad = np.load(individual_rank_path)

# Wang: overlap-weighted aggregation of each subject's regional gradient (see wang_utils.py)
def build_rank_wang_df(indiv_grad, merge_vd):
    areas = wu.wang_areas_merged if merge_vd else wu.wang_areas
    rows = []
    for j in range(indiv_grad.shape[1]):
        col = indiv_grad[:, j]
        if np.all(np.isnan(col)):
            continue
        agg = wu.aggregate_to_wang(col, merge_vd=merge_vd, min_area_vox=1)
        for a in areas:
            rows.append({'subject': j + 1, 'area': a, 'mean_rank': agg[a]})
    return pd.DataFrame(rows)

rank_wang_df = build_rank_wang_df(indiv_syn_red_grad, merge_vd=True)
rank_wang_split_df = build_rank_wang_df(indiv_syn_red_grad, merge_vd=False)
rank_wang_df.to_csv(f'{main_dir}/{parcellation}/indiv_rank_wang.csv', index=False)
rank_wang_split_df.to_csv(f'{main_dir}/{parcellation}/indiv_rank_wang_split.csv', index=False)

# orderings
order_yeo = [network_dict[n] for n in network_dict]
order_mesulam = ['Idiotypic', 'Unimodal', 'Heteromodal', 'Paralimbic']
order_wang_streams = ['V1', 'V2', 'V3', 'V4', 'VO1', 'VO2', 'PHC1', 'PHC2',
                      'LO1', 'LO2', 'V3a', 'V3b', 'hMT', 'MST',
                      'IPS/SPL', 'FEF']
order_wang_streams_split = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4',
                            'VO1', 'VO2', 'LO1', 'LO2', 'V3a', 'V3b', 'hMT', 'MST']

# one-sample t vs 0, BH-corrected within each grouping (over the areas actually shown)
def rank_stats(df, cat, order):
    rows = []
    for name in order:
        vals = df[df[cat] == name]['mean_rank'].values
        vals = vals[np.isfinite(vals)]
        t, p, cil, ciu, d, bf, dof, power = misc.aug_t_test(vals, 0, paired=False)
        rows.append({cat: name, 't': t, 'p': p, 'cil': cil, 'ciu': ciu,
                     'd': d, 'bf': bf, 'dof': dof, 'power': power})
    out = pd.DataFrame(rows)
    out['p_bh'] = multi.multipletests(out['p'], method='fdr_bh')[1]
    return out

rank_yeo_stats        = rank_stats(rank_yeo_df, 'network', order_yeo)
rank_mesulam_stats    = rank_stats(rank_mesulam_df, 'type', order_mesulam)
rank_wang_stats       = rank_stats(rank_wang_df, 'area', order_wang_streams)
rank_wang_split_stats = rank_stats(rank_wang_split_df, 'area', order_wang_streams_split)

for (df_s, cat, order) in [(rank_wang_stats, 'area', order_wang_streams),
                           (rank_yeo_stats, 'network', order_yeo),
                           (rank_mesulam_stats, 'type', order_mesulam)]:
    for name in order:
        row = df_s[df_s[cat] == name].iloc[0]
        t = misc.write_scientific(row['t'])
        p = misc.write_scientific(row['p_bh'])
        print(f'{name}: $T=${t}, $p_{{FDR}}=${p}')


# %%
# Paired contrasts across the retinotopic streams, reported in the Results
from scipy.stats import spearmanr

parietal_fields = ['IPS0', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'SPL1']

def area_wide(df):
    return df.pivot(index='subject', columns='area', values='mean_rank')

def paired_contrast(name, a, b):
    keep = np.isfinite(a) & np.isfinite(b)
    t, p, cil, ciu, d, bf, dof, power = misc.aug_t_test(a[keep], b[keep], paired=True)
    print(f'{name}: {a[keep].mean():+.1f} against {b[keep].mean():+.1f}, '
          f'$T=${misc.write_scientific(t)}, $p=${misc.write_scientific(p)}, '
          f'$d={d:.2f}$, $BF=${misc.write_scientific(bf)}')

wang_wide = area_wide(rank_wang_df)
split_wide = area_wide(rank_wang_split_df)
motion = wang_wide[['hMT', 'MST']].mean(axis=1).values

paired_contrast('hMT/MST against IPS/SPL', motion, wang_wide['IPS/SPL'].values)
paired_contrast('hMT/MST against FEF', motion, wang_wide['FEF'].values)
paired_contrast('early dorsal against early ventral',
                split_wide[['V1d', 'V2d', 'V3d']].mean(axis=1).values,
                split_wide[['V1v', 'V2v', 'V3v']].mean(axis=1).values)

# the seven parietal fields are pooled only because they carry no reliable ordering
ladder = split_wide[parietal_fields].values
rho = np.array([spearmanr(np.arange(len(parietal_fields)), row).correlation
                for row in ladder])
rho = rho[np.isfinite(rho)]
t, p, cil, ciu, d, bf, dof, power = misc.aug_t_test(rho, 0, paired=False)
print(f'ordering within IPS0-SPL1: rho = {rho.mean():+.3f}, '
      f'$p=${misc.write_scientific(p)}, $BF=${misc.write_scientific(bf)}')


# %%
# Correlation between each subject's regional syn-red gradient and the group-level one

from scipy.stats import spearmanr

group_grad = syn_red_grad  # group-level gradient (computed earlier)

def grad_corr(a, b):
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return np.nan
    return spearmanr(a[mask], b[mask]).correlation

# whole-brain: one correlation per subject
wholebrain_corrs = np.array([grad_corr(indiv_syn_red_grad[:, j], group_grad)
                             for j in range(indiv_syn_red_grad.shape[1])])
wholebrain_corrs = wholebrain_corrs[~np.isnan(wholebrain_corrs)]

print(f'Individual vs group gradient: '
      f'r = {np.mean(wholebrain_corrs):.3f} +/- {np.std(wholebrain_corrs):.3f}')


fig, ax = plt.subplots(figsize=(1.75, 1.4))
sns.violinplot(x=wholebrain_corrs, ax=ax, orient='h', 
               split=True, inner=None, color='lightgrey')
ax.set_xlabel(r'Sim. to group')
ax.set_yticks([])
ax.set_xlim(0, 1)
ax.xaxis.set_major_locator(plt.MaxNLocator(2))
sns.despine(left=True)
plt.tight_layout()
plt.savefig(f'{plots_dir}/indiv_group_synred_corr_hist.png', bbox_inches='tight')
plt.show()


# %%
# Box plots: individual-level synergy-redundancy rank by Wang area, Yeo network, Mesulam type
clim = 460 
norm = plt.Normalize(-clim, clim)

def draw_rank_panel(ax, df, cat, order, stats, first=False):
    for i, name in enumerate(order):
        nd = df[df[cat] == name]['mean_rank'].values
        nd = nd[np.isfinite(nd)]
        ax.scatter(np.full(len(nd), i) + np.random.normal(0, 0.05, len(nd)),
                   nd, c=misc.sr_full_cmap(norm(nd)), alpha=0.7, zorder=5)
    sns.boxplot(x=cat, y='mean_rank', data=df, order=order, ax=ax,
                showcaps=False, showfliers=False,
                boxprops={'facecolor': 'none', 'zorder': 10, 'edgecolor': 'black', 'linewidth': 1},
                medianprops={'color': 'black', 'zorder': 11, 'linewidth': 1},
                whiskerprops={'color': 'black', 'zorder': 11, 'linewidth': 1},
                width=0.6)
    ax.set_ylim(-clim, clim)
    ax.axhline(0, color='grey', linestyle='--', zorder=0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90)
    ax.set_xlabel('')
    if first:
        ax.set_ylabel('Rank', labelpad=-20)
    else:
        ax.set_ylabel('')
        ax.set_yticks([])
    for i, name in enumerate(order):
        row = stats[stats[cat] == name].iloc[0]
        c = '#4f062d' if row['t'] > 0 else '#1D5B4A'
        star = misc.get_star(row['p_bh']) if row['p_bh'] < 0.05 else ''
        ax.text(i, clim * 0.9, star, ha='center', va='bottom', color=c, fontsize=12)

fig, ax = plt.subplots(1, 3, figsize=(8.5, 4),
                       gridspec_kw={'width_ratios': [12, 13.5, 3]})

draw_rank_panel(ax[0], rank_wang_df, 'area', order_wang_streams, rank_wang_stats, first=True)
draw_rank_panel(ax[1], rank_yeo_df, 'network', order_yeo, rank_yeo_stats)
draw_rank_panel(ax[2], rank_mesulam_df, 'type', order_mesulam, rank_mesulam_stats)

sns.despine()
for a in ax[1:]:
    a.spines['left'].set_visible(False)

ax[0].set_yticks([-250, 0, 250])


ax[1].get_xticklabels()[-1].set_color('darkgrey')

plt.tight_layout()
plt.subplots_adjust(wspace=0.025)
plt.savefig(f'{plots_dir}/indiv_synred_rank_wang_yeo_mesulam.png', bbox_inches='tight')
plt.show()


# %%
## FC corr
#
#from scipy.stats import spearmanr
#
#def best_quadrant_pos(ax, x, y, pad=0.05):
#    """Returns (x, y) axes-fraction coords for text in the emptiest quadrant."""
#    x_mid = (ax.get_xlim()[0] + ax.get_xlim()[1]) / 2
#    y_mid = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2
#
#    quadrants = {
#        'TL': (x < x_mid) & (y > y_mid),
#        'TR': (x > x_mid) & (y > y_mid),
#        'BL': (x < x_mid) & (y < y_mid),
#        'BR': (x > x_mid) & (y < y_mid),
#    }
#    emptiest = min(quadrants, key=lambda q: quadrants[q].sum())
#
#    corners = {
#        'TL': (pad,     1 - pad),
#        'TR': (1 - pad, 1 - pad),
#        'BL': (pad,     pad),
#        'BR': (1 - pad, pad),
#    }
#    ha = 'left' if emptiest.endswith('L') else 'right'
#    va = 'top'  if emptiest.startswith('T') else 'bottom'
#    return (*corners[emptiest], ha, va)
#
#cmap = cmr.amber
#corr_stats_df = pd.DataFrame(columns=['atom', 'rho', 'p'])
#
#panels = []
#for atom in misc.atom_order:
#    atom_indices = misc.atoms[atom]
#    info_mats = [m for m in (load_connectome(perp, atom_indices, fix_nans=False) for perp in subjects) if m is not None]
#    fc_mats = [m for m in (load_fc(perp, fix_nans=False) for perp in subjects) if m is not None]
#    n = info_mats[0].shape[0]
#    tri_idx = np.triu_indices(n, k=1)
#    x = np.concatenate([m[tri_idx] for m in info_mats]).flatten()
#    y = np.concatenate([m[tri_idx] for m in fc_mats]).flatten()
#    nan_idx = np.isnan(x) | np.isnan(y)
#    x, y = x[~nan_idx], np.abs(y[~nan_idx])
#    rho, p = spearmanr(x, y)
#    corr_stats_df = pd.concat([corr_stats_df, pd.DataFrame({'atom': [atom], 'rho': [rho], 'p': [p]})], ignore_index=True)
#    panels.append((misc.atoms_names[atom], x, y, rho))
#
## TDMI
#info_mats = [m for m in (np.sum(np.load(f'{main_dir}/{parcellation}/sub-{perp}/phiid_matrix.npy'), axis=-1) for perp in subjects) if m is not None]
#fc_mats = [m for m in (load_fc(perp, fix_nans=False) for perp in subjects) if m is not None]
#n = info_mats[0].shape[0]
#tri_idx = np.triu_indices(n, k=1)
#x = np.concatenate([m[tri_idx] for m in info_mats]).flatten()
#y = np.concatenate([m[tri_idx] for m in fc_mats]).flatten()
#nan_idx = np.isnan(x) | np.isnan(y)
#x, y = x[~nan_idx], np.abs(y[~nan_idx])
#rho_tdmi, _ = spearmanr(x, y)
#panels.append((r'$TDMI$', x, y, rho_tdmi))
#
#corr_stats_df['p_bh'] = multi.multipletests(corr_stats_df['p'], method='fdr_bh')[1]
#corr_stats_df.to_csv(f'{main_dir}/{parcellation}/FC_corr_stats.csv', index=False)
#for atom in misc.atom_order:
#    rho = misc.write_scientific(corr_stats_df.loc[corr_stats_df['atom']==atom, 'rho'].values[0])
#    p = misc.write_scientific(corr_stats_df.loc[corr_stats_df['atom']==atom, 'p_bh'].values[0])
#    print(f'{misc.atoms_names[atom]}: $r=${rho};', f'$p_{{FDR}}=${p}')
#
#fig, axes = plt.subplots(3, 4, figsize=(14, 10.5))
#axes = axes.flatten()
#
#for i, (label, x, y, rho) in enumerate(panels):
#    ax = axes[i]
#    sns.histplot(y=y, x=x, bins=1000, cmap=cmap, vmax=200, ax=ax)
#    ax.set_xlabel(label)
#    ax.set_ylabel(r'$|FC|$' if i % 4 == 0 else '')
#    tx, ty, ha, va = best_quadrant_pos(ax, x, y)
#    ax.text(tx, ty, fr'$r$ = {rho:.2f}', transform=ax.transAxes, ha=ha, va=va)
#    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
#    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
#    sns.despine(ax=ax)
#
#axes[-1].set_visible(False)
#
#plt.tight_layout()
#plt.savefig(f'{plots_dir}/FC_corr_all.png', bbox_inches='tight')
#plt.show()


# %%
from scipy.stats import spearmanr
from tqdm import tqdm

fc_keys = list(misc.atom_order) + ['TDMI']
fc_cache = f'{main_dir}/{parcellation}/FC_corr_hist.npz'
fc_stats_path = f'{main_dir}/{parcellation}/FC_corr_stats.csv'
n_bins = 400

if not os.path.exists(fc_cache):
    triu = np.triu_indices(n_regions, k=1)
    fc_x = {k: [] for k in fc_keys}
    fc_y = []
    for perp in tqdm(subjects, desc='FC vs atoms'):
        mat = np.load(f'{main_dir}/{parcellation}/sub-{perp}/phiid_matrix.npy'
                      ).astype(np.float64)
        fc = np.abs(np.load(f'{main_dir}/{parcellation}/sub-{perp}/pearson_matrix.npy'))
        keep = (np.isfinite(fc[triu])
                & np.all(np.isfinite(mat[triu[0], triu[1], :]), axis=1))
        fc_y.append(fc[triu][keep].astype(np.float32))
        for atom in misc.atom_order:
            fc_x[atom].append(
                mat[:, :, misc.atoms[atom]].sum(axis=2)[triu][keep].astype(np.float32))
        fc_x['TDMI'].append(mat.sum(axis=2)[triu][keep].astype(np.float32))
        del mat, fc

    fc_y = np.concatenate(fc_y)
    fc_x = {k: np.concatenate(v) for k, v in fc_x.items()}
    print(f'{len(fc_y):,} edge observations over {len(subjects)} participants')

    fc_rows = []
    hist_store = {}
    y_hi = np.percentile(fc_y, 99.95)
    for k in tqdm(fc_keys, desc='Spearman'):
        rho, p = spearmanr(fc_x[k], fc_y)
        fc_rows.append({'atom': k, 'rho': rho, 'p': p})
        # 2D histogram rather than a scatter: at half a billion points a scatter is
        # both unreadable and unplottable
        x_lo, x_hi = np.percentile(fc_x[k], [0.05, 99.95])
        counts, xe, ye = np.histogram2d(fc_x[k], fc_y, bins=n_bins,
                                        range=[[x_lo, x_hi], [0, y_hi]])
        hist_store[f'counts_{k}'] = counts.astype(np.float32)
        hist_store[f'extent_{k}'] = np.array([xe[0], xe[-1], ye[0], ye[-1]])

    fc_stats = pd.DataFrame(fc_rows)
    fc_stats['p_bh'] = multi.multipletests(fc_stats['p'], method='fdr_bh')[1]
    fc_stats.to_csv(fc_stats_path, index=False)
    np.savez_compressed(fc_cache, n_obs=len(fc_y), **hist_store)
    del fc_x, fc_y
else:
    hist_store = np.load(fc_cache)
    fc_stats = pd.read_csv(fc_stats_path)
    print(f"{int(hist_store['n_obs']):,} edge observations over {len(subjects)} participants")

fc_labels = {**{a: misc.atoms_names[a] for a in misc.atom_order}, 'TDMI': r'$\mathrm{TDMI}$'}
ncol = 4
nrow = int(np.ceil(len(fc_keys) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.0 * nrow))
axes = axes.flatten()

for ax, k in zip(axes, fc_keys):
    counts = np.asarray(hist_store[f'counts_{k}'])
    extent = np.asarray(hist_store[f'extent_{k}'])
    # counts are clipped so the mode of the joint distribution does not flatten
    # everything else to a single colour
    ax.imshow(counts.T, origin='lower', aspect='auto', cmap=cmr.amber,
              extent=extent, vmax=np.percentile(counts[counts > 0], 99))
    rho = float(fc_stats.loc[fc_stats['atom'] == k, 'rho'].iloc[0])
    ax.text(0.04, 0.96, fr'$\rho$ = {rho:.2f}', transform=ax.transAxes,
            ha='left', va='top', color='white')
    ax.set_xlabel(fc_labels[k])
    ax.set_ylabel(r'$|FC|$')
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    sns.despine(ax=ax)

for ax in axes[len(fc_keys):]:
    ax.set_visible(False)

plt.tight_layout()
fig.savefig(f'{plots_dir}/FC_correlation.png', bbox_inches='tight')
plt.show()

#for k in fc_keys:
#    r = fc_stats[fc_stats['atom'] == k].iloc[0]
#    print(f"{fc_labels[k]}: $\\rho=${misc.write_scientific(r['rho'])}, "
#          f"{misc.write_p(r['p_bh'], sub='FDR')}")
