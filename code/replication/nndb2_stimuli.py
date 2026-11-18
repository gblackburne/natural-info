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
font_dir = '/home/george/Research/fonts/Instrument_Sans/static'
import glob
_have = {(f.name, f.weight, f.style) for f in fm.fontManager.ttflist}
for _f in sorted(glob.glob(f'{font_dir}/InstrumentSans-*.ttf')):
    try:
        _p = fm.FontProperties(fname=_f)
        if (_p.get_name(), _p.get_weight(), _p.get_style()) not in _have:
            fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams['font.family'] = 'Instrument Sans'
plt.rcParams['font.family'] = 'Instrument Sans'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 800
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = "cm"
rc_fontsize = 15
plt.rcParams.update({'font.size': rc_fontsize})

root_dir = os.path.abspath('..')
if root_dir not in sys.path:
    sys.path.append(root_dir)  # network_utils and the root stimuli modules live there
# misc still resolves to replication/misc.py: the notebook's own directory comes first

import network_utils as nu
import misc

parcellation = misc.parcellation
n_regions = misc.n_regions

MRI_path = '/home/george/Research/fMRI_general'
atlas_info = pd.read_csv(f'{MRI_path}/atlases/schaefer_extend_atlases/{parcellation}.csv')

main_dir = f'nndb2_results/main'
stimuli_dir = f'nndb2_results/stimuli'
orig_stimuli_dir = f'../results/stimuli'  # the published run, read only
if not os.path.exists(stimuli_dir):
    os.makedirs(stimuli_dir)

plots_dir = f'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

modes = misc.atom_order

from palettable.scientific.diverging import Berlin_20
berlin_cmap = Berlin_20.mpl_colormap

recompute = False  # set True to force any cached analysis below to recompute

norm_dir = f'{stimuli_dir}/normative'
mvf_dir = f'{stimuli_dir}/embeddings'
orig_norm_dir = f'{orig_stimuli_dir}/normative'
orig_mvf_dir = f'{orig_stimuli_dir}/embeddings'
n_nulls = 10000


# %%
# check that resmp q looks okkkkk
import nndb2_stimuli_utils as su

n_ann = su.load_embeddings(su.movie).shape[0]
_, seqs = su.load_winner_states()
seq_len = len(next(iter(seqs.values())))
stack = su.build_stack(offset=-5)
mv = stack[su.movie]
span_len = np.array([len(r) for r in mv['cap_rows']])
single = su.build_stack(offset=-5, caption_mode='single')[su.movie]

fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))

# two timelines over the first half minute
ax = axes[0]
n_show = 20
for i in range(n_show):
    ax.add_patch(plt.Rectangle((i * su.annotation_tr, 1.05), su.annotation_tr * 0.96, 0.32,
                               facecolor='0.75', edgecolor='black', lw=0.4))
for i in range(int(n_show / su.tr_ratio)):
    ax.add_patch(plt.Rectangle((i * su.scan_tr, 0.63), su.scan_tr * 0.96, 0.32,
                               facecolor=misc.atoms_colors['Preservation'],
                               edgecolor='black', lw=0.4))
for i in range(int(n_show / su.tr_ratio)):
    ax.plot([i * su.scan_tr + 0.05, np.floor(i * su.tr_ratio) + 0.5], [0.63, 1.05],
            color='0.4', lw=0.7, zorder=0)
ax.set_xlim(0, n_show); ax.set_ylim(0.5, 1.5)
ax.set_yticks([0.79, 1.21]); ax.set_yticklabels(['Atom window\n1.5 s', 'Annotation\n1 s'],
                                                fontsize=rc_fontsize)
ax.set_xlabel('Movie time (s)')
ax.set_title('Alignment', fontsize=rc_fontsize, loc='left')
for sp in ('top', 'right', 'left'):
    ax.spines[sp].set_visible(False)
ax.tick_params(axis='y', length=0)

# ow much of the annotation grid eachmode reaches
ax = axes[1]
reach = [('Annotation \n rows', n_ann),
         ('Anchor \n rows used', len(np.unique(mv['emb_rows']))),
         ('Rows any \n span touches', len(np.unique(np.concatenate(mv['cap_rows']))))]
ax.barh(np.arange(len(reach)), [v for _, v in reach], color=['0.75', '0.45', '0.25'],
        edgecolor='black', lw=0.6)
for i, (_, v) in enumerate(reach):
    ax.text(v + n_ann * 0.02, i, f'{v:,}', va='center', fontsize=rc_fontsize, color='0.3')
ax.set_yticks(np.arange(len(reach))); ax.set_yticklabels([k for k, _ in reach],
                                                         fontsize=rc_fontsize)
ax.invert_yaxis(); ax.set_xlim(0, n_ann * 1.22)
ax.set_xlabel('Annotation rows')
ax.set_title('Grid coverage', fontsize=rc_fontsize, loc='left')
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)

# what the span buys on the text side
ax = axes[2]
names_n, dicts_n = su.load_norms(su.norm_manifest)
f_span, tok_span = su.build_norm_features(su.movie, mv['cap_rows'], names_n, dicts_n, 'all')
f_one, tok_one = su.build_norm_features(su.movie, single['cap_rows'], names_n, dicts_n, 'all')
bins = np.linspace(0, max(tok_span.max(), tok_one.max()), 40)
ax.hist(tok_one, bins=bins, color='0.6', edgecolor='black', lw=0.4, label='Single row')
ax.hist(tok_span, bins=bins, color=misc.atoms_colors['Cooperation'], alpha=0.75,
        edgecolor='black', lw=0.4, label='Span')
ax.set_xlabel('Tokens per atom window'); ax.set_ylabel('Windows')
ax.set_title('Text per window', fontsize=rc_fontsize, loc='left')
ax.legend(frameon=False, fontsize=rc_fontsize)
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
ax.yaxis.set_major_locator(plt.MaxNLocator(3))
sns.despine(ax=ax)

for lab, ax in zip('abc', axes):
    ax.text(-0.06, 1.12, lab, transform=ax.transAxes, fontsize=rc_fontsize,
            va='top', ha='right', fontweight='bold')
plt.tight_layout()
fig.savefig(f'{plots_dir}/replication_resampling_QC.png', bbox_inches='tight', transparent=True)
plt.show()

print(f'{seq_len:,} atom windows at {su.scan_tr} s against {n_ann:,} annotation rows at '
      f'{su.annotation_tr} s ({seq_len * su.scan_tr:,.1f} s vs {n_ann * su.annotation_tr:,.1f} s, '
      f'{abs(seq_len * su.scan_tr - n_ann * su.annotation_tr):.1f} s apart)')
print(f"{len(mv['emb_rows']):,} windows survive the {abs(-5)} s lag and the validity mask, "
      f'each anchored on one row and spanning '
      f'{span_len.min()} to {span_len.max()} rows')
print(f'tokens per window: {tok_one.mean():.1f} single row, {tok_span.mean():.1f} span '
      f'({tok_span.mean() / tok_one.mean() - 1:+.0%})')


# %%
import json

def norm_label(nm):
    special = {'log_frequency': 'Word density', 'hand_arm': 'Hand/arm',
               'foot_leg': 'Foot/leg', 'body_action': 'Action',
                'social_interaction': 'Interaction'}
    return special.get(nm, nm.replace('_', ' ').capitalize())

norm_corr = pd.read_csv(f'{stimuli_dir}/normative/norm_correlations.csv', index_col=0)
norm_summary = json.load(open(f'{stimuli_dir}/normative/summary.json'))
ring = norm_summary['norm_ring_order']
assert set(ring) == set(norm_corr.columns), 'ring order and correlation matrix disagree'

corr_ring = norm_corr.loc[ring, ring]
labels_ring = [norm_label(nm) for nm in ring]
n_ring = len(ring)

fig, ax = plt.subplots(figsize=(6, 5.4))

vlim = 1
im = ax.imshow(corr_ring.values, cmap=berlin_cmap, vmin=-vlim, vmax=vlim)
ax.set_xticks(range(n_ring))
ax.set_xticklabels(labels_ring, rotation=90, fontsize=rc_fontsize - 4)
ax.set_yticks(range(n_ring))
ax.set_yticklabels(labels_ring, fontsize=rc_fontsize - 4)
ax.set_xticks(np.arange(n_ring + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(n_ring + 1) - 0.5, minor=True)
ax.grid(which='minor', color='black', linewidth=0.4)
ax.tick_params(which='minor', length=0)

for spine in ax.spines.values():
    spine.set_visible(False)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.ax.yaxis.set_major_locator(plt.MaxNLocator(3))
cbar.ax.set_title('$r$', pad=8)

plt.tight_layout()
plt.show()

off = corr_ring.values[~np.eye(n_ring, dtype=bool)]
print(f'off-diagonal |r|: max = {np.abs(off).max():.3f}, '
      f'mean = {np.abs(off).mean():.3f}')
# correlation of each spoke with its clockwise neighbour, ie the joins the seriation
# maximised. the wrap join is the weakest one, rotated there by ring_order()
adj = np.array([corr_ring.values[i, (i + 1) % n_ring] for i in range(n_ring)])
print(f'mean neighbour join r = {adj.mean():.3f}, '
      f'weakest (wrap) join = {adj.min():.3f}')
iu = np.triu_indices(n_ring, k=1)
top = np.argsort(np.abs(corr_ring.values[iu]))[::-1][:5]
print('strongest pairs:')
for k in top:
    i, j = iu[0][k], iu[1][k]
    print(f'  {labels_ring[i]} / {labels_ring[j]}: r = {corr_ring.values[i, j]:.3f}')


# %%
from matplotlib.colors import LinearSegmentedColormap
norms_df = pd.read_csv('nndb2_results/stimuli/normative/feature_atom_zscores.csv')

fdr_sub = r'\mathrm{FDR}'

sig_thresh = 0.05
sig_col = 'q_fdr'  # 'q_fdr' (FDR-corrected) or 'p_perm'
norms_df['is_sig'] = norms_df[sig_col] < sig_thresh

mode_seq = ['Oneness', 'Encryption', 'Dissolution', 'Copy', 'Migration',
            'Erasure', 'Unification', 'Preservation', 'Decryption', 'Cooperation']

ring_order = json.load(open(f'{orig_norm_dir}/summary.json'))['norm_ring_order']
ring_rot_deg = 0

n_norm = len(ring_order)
z_by = norms_df.pivot(index='name', columns='feature', values='z')
sig_by = norms_df.pivot(index='name', columns='feature', values='is_sig')
norm_ang = {nm: (np.pi / 2 + np.radians(ring_rot_deg) - 2 * np.pi * i / n_norm) % (2 * np.pi)
            for i, nm in enumerate(ring_order)}
zclip = 6.0
r0 = 10.0
rmax = r0 + zclip
bar_w = 2 * np.pi / n_norm * 0.8

key_grad = LinearSegmentedColormap.from_list('key_grad', ['white', '0.6'])


def style_coin(ax):
    ax.set_ylim(0, rmax); ax.set_xticks([]); ax.set_yticks([])
    ax.spines['polar'].set_visible(False); ax.set_facecolor('none')
    # fait guide from the origi out to
    for a in norm_ang.values():
        ax.plot([a, a], [0, r0], color='0.85', lw=0.4, zorder=0)
    ax.plot(np.linspace(0, 2 * np.pi, 200), np.full(200, r0), color='0.55', lw=1.0, zorder=2)


def draw_bars(ax, atom):
    col = misc.atoms_colors[atom]
    for nm in ring_order:
        z = float(z_by.loc[atom, nm])
        h = min(abs(z), zclip)
        bottom = r0 if z >= 0 else r0 - h
        if bool(sig_by.loc[atom, nm]):
            ax.bar(norm_ang[nm], h, width=bar_w, bottom=bottom, color=col,
                   edgecolor='black', linewidth=0.5, zorder=3)
        else:
            ax.bar(norm_ang[nm], h, width=bar_w, bottom=bottom, color='0.5',
                   alpha=0.15, linewidth=0, zorder=1)


fig = plt.figure(figsize=(14, 4))
#fig.patch.set_facecolor('none')
axk = fig.add_axes([-.03, 0.175, 0.4, 0.6], projection='polar'); style_coin(axk)
gt, gr = np.meshgrid(np.linspace(0, 2 * np.pi, 240), np.linspace(0, rmax, 120))
for nm in ring_order:
    ang = norm_ang[nm]
    deg = np.degrees(ang) % 360
    flip = 90 < deg < 270
    axk.text(ang, r0, (norm_label(nm) + '  ' if flip else '  ' + norm_label(nm)),
             rotation=(deg + 180 if flip else deg), rotation_mode='anchor',
             ha=('right' if flip else 'left'), va='center', fontsize=rc_fontsize - 1, color='0.2')

gx0, gx1, gy0, gy1 = 0.3, 0.99, 0, 0.95
ncol, nrow = 5, 2
coin_scale = 1.1  # alloww overlap
cw = (gx1 - gx0) / ncol
ch = (gy1 - gy0) / nrow
fig_w, fig_h = fig.get_size_inches()
size = min(cw * fig_w, ch * fig_h) / fig_w * coin_scale  # keeps coins circul
size_h = size * (fig_w / fig_h)
for idx, atom in enumerate(mode_seq):
    r, c = divmod(idx, ncol)
    cx = gx0 + (c + 0.5) * cw
    cy = gy1 - (r + 0.5) * ch
    ax = fig.add_axes([cx - size / 2, cy - size_h / 2, size, size_h], projection='polar')
    ax.patch.set_alpha(0)
    style_coin(ax)
    draw_bars(ax, atom)
    ax.text(0, 0, misc.atoms_names[atom], color=misc.atoms_colors[atom],
            fontsize=rc_fontsize, ha='center', va='center')

fig.savefig(f'{plots_dir}/atom_norm_coins.png', bbox_inches='tight', transparent=True)

for atom in mode_seq:
    sub = norms_df[(norms_df['name'] == atom) & (norms_df['is_sig'])]
    sub = sub.reindex(sub['z'].abs().sort_values(ascending=False).index)
    print(f'{atom}  ({len(sub)} sig)')
    for _, row in sub.iterrows():
        print(f"    {norm_label(row['feature']):16s} "
              f"z={misc.write_scientific(row['z'])}, "
              f"{misc.write_p(row['p_perm'], floor=1e-4, sub='perm')}, "
              f"{misc.write_p(row['q_fdr'], floor=1e-4, sub=fdr_sub)}")

own = json.load(open(f'{norm_dir}/summary.json'))
print()
print(f"ring order held at the published one; nndb2's own seriation smoothness "
      f"{own['norm_ring_smoothness']:.3f}")
print('  its own order: ' + ', '.join(norm_label(n) for n in own['norm_ring_order']))


# %%
from matplotlib.patches import Rectangle, Polygon

fig = plt.figure(figsize=(8.5, 0.8))

lr0 = 24.0
lbw = np.radians(5)
lbh = 3.0
th0, th1 = np.radians(77), np.radians(107)
th_ruler = np.radians(104)

def arc_bar(ax, ac, rin, rout, **kw):
    a = np.linspace(ac - lbw / 2, ac + lbw / 2, 30)
    pts = np.concatenate([np.c_[rin * np.cos(a), rin * np.sin(a)],
                          np.c_[rout * np.cos(a[::-1]), rout * np.sin(a[::-1])]])
    ax.add_patch(Polygon(pts, closed=True, **kw))


axz = fig.add_axes([0.0, 0.0, 0.12, 1.0])
axz.set_aspect('equal'); axz.set_axis_off()
a = np.linspace(th0, th1, 200)
axz.plot(lr0 * np.cos(a), lr0 * np.sin(a), color='0.55', lw=1.0, zorder=2)
arc_bar(axz, np.radians(93), lr0, lr0 + lbh, facecolor='0.75',
        edgecolor='black', linewidth=0.5, zorder=3)
arc_bar(axz, np.radians(85), lr0 - lbh, lr0, facecolor='0.75',
        edgecolor='black', linewidth=0.5, zorder=3)

rad = np.array([np.cos(th_ruler), np.sin(th_ruler)])
tan = np.array([-np.sin(th_ruler), np.cos(th_ruler)])
ends = np.array([(lr0 - zclip) * rad, (lr0 + zclip) * rad])
axz.plot(ends[:, 0], ends[:, 1], color='0.2', lw=0.8, zorder=2)
for zt in np.arange(-zclip, zclip + 1e-9, 2):
    p = (lr0 + zt) * rad
    major = zt in (-zclip, 0, zclip)
    q = p + tan * (1.3 if major else 0.7)
    axz.plot([p[0], q[0]], [p[1], q[1]], color='0.2', lw=0.8, zorder=2)
    if major:
        lab = p + tan * 2.0
        axz.text(lab[0], lab[1], f'{zt:+.0f}'.replace('+0', '0'), ha='right',
                 va='center', fontsize=rc_fontsize - 3, color='0.2')
ztop = (lr0 + zclip + 1.6) * rad
axz.text(ztop[0], ztop[1], '$z$', ha='center', va='center',
         fontsize=rc_fontsize - 2, color='0.2')
axz.set_xlim(-12, 6); axz.set_ylim(lr0 - zclip - 1.2, lr0 + zclip + 2.6)

fig.text(0.125, 0.64, r'out: above null ($z > 0$)', va='center', fontsize=rc_fontsize - 1)
fig.text(0.125, 0.26, r'in: below null ($z < 0$)', va='center', fontsize=rc_fontsize - 1)

def swatch(x, w=0.028, h=0.3):
    ax = fig.add_axes([x, 0.5 - h / 2, w, h])
    ax.set_xticks([]); ax.set_yticks([])
    return ax

axs = swatch(0.4)
for i, atom in enumerate(mode_seq):
    axs.add_patch(Rectangle((i / len(mode_seq), 0), 1 / len(mode_seq), 1,
                            facecolor=misc.atoms_colors[atom], edgecolor='none'))
for sp in axs.spines.values():
    sp.set_linewidth(0.8)
fig.text(0.44, 0.5, r'$p_{\mathrm{FDR}} < \alpha$', va='center', fontsize=rc_fontsize)

axn = swatch(0.575)
axn.add_patch(Rectangle((0, 0), 1, 1, facecolor='0.5', alpha=0.15, linewidth=0))
for sp in axn.spines.values():
    sp.set_color('0.7'); sp.set_linewidth(0.8)
fig.text(0.615, 0.5, 'n.s.', va='center', fontsize=rc_fontsize)

fig.savefig(f'{plots_dir}/atom_norm_coins_legend.png', bbox_inches='tight', transparent=True)
plt.show()


# %%
import json
norm_dir = 'nndb2_results/stimuli/normative'
nsum = json.load(open(f'{norm_dir}/summary.json'))
ndf = pd.read_csv(f'{norm_dir}/feature_atom_zscores.csv')
nsig = ndf[ndf['q_fdr'] < 0.05]
eff = nsig['mean_sd_units'].abs()

print(f"{len(nsig)}/{len(ndf)} atom x norm cells survive FDR ({len(nsig) / len(ndf):.0%}), "
      f"over {nsig['name'].nunique()}/10 modes")
print(f"sample: {nsum['n_participants']} participants, all watching "
      f"{'one film' if nsum['n_films'] == 1 else str(nsum['n_films']) + ' films'}, "
      f"{nsum['n_occurrences']:,} atom occurrences, {len(nsum['norm_dimensions'])} norm dims")
print(f"\neffects:: biggest {eff.max():.3f} SD, median {eff.median():.3f} SD")

for mode in mode_seq:
    sub = nsig[nsig['name'] == mode]
    sub = sub.reindex(sub['z'].abs().sort_values(ascending=False).index).head(4)
    if sub.empty:
        print(f'{mode}: nothing survived FDR')
        continue
    bits = [f"{r['feature']} ({'higher' if r['z'] > 0 else 'lower'}, "
            f"{r['mean_sd_units']:+.3f} SD, $z$={r['z']:.2f}, "
            f"{misc.write_p(r['q_fdr'], floor=2e-4, sub=fdr_sub)})"
            for _, r in sub.iterrows()]
    print(f'{mode}: ' + '; '.join(bits))
# probs say v reliably non-random but small, not large semantic segregation


# %%
# replicating the norm profiles
# the atom x norm z table is the whole result here, 
# so just kep porofiles intact and just relabels the modes.

def z_table(path):
    d = pd.read_csv(path)
    return d.pivot(index='name', columns='feature', values='z')

o_norm = z_table(f'{orig_norm_dir}/feature_atom_zscores.csv')
r_norm = z_table(f'{norm_dir}/feature_atom_zscores.csv')
s_norm = z_table(f'{norm_dir}/caption_single/feature_atom_zscores.csv')
assert list(o_norm.columns) == list(r_norm.columns) == list(s_norm.columns), 'norm dims disagree'
norm_cols = list(o_norm.columns)

norm_csv = f'{norm_dir}/replication_norms.csv'
if os.path.exists(norm_csv) and not recompute:
    norm_rep = pd.read_csv(norm_csv)
else:
    head = []
    for label, a, b in [('Norm profiles (span)', o_norm, r_norm),
                        ('Norm profiles (single)', o_norm, s_norm),
                        ('Caption span vs single', r_norm, s_norm)]:
        res = misc.profile_null_corr(a.loc[modes, norm_cols].values,
                                     b.loc[modes, norm_cols].values, n_perm=n_nulls)
        head.append({'block': 'headline', 'comparison': label, 'r': res['r'], 'p': res['p'],
                     'n_items': res['n_items']})
    # then each mode on its own, permuting the norms within that mode
    per = []
    for m in modes:
        res = misc.perm_corr(o_norm.loc[m, norm_cols].values,
                             r_norm.loc[m, norm_cols].values, n_perm=n_nulls)
        per.append({'block': 'per_mode', 'comparison': m, 'r': res['r'], 'p': res['p'],
                    'n_items': res['n_items']})
    per = pd.DataFrame(per)
    per['p_bh'] = multi.multipletests(per['p'].values, method='fdr_bh')[1]
    # the three headline rows ask three different questions and are not a family
    norm_rep = pd.concat([pd.DataFrame(head), per], ignore_index=True)
    norm_rep.to_csv(norm_csv, index=False)

for _, r in norm_rep[norm_rep['block'] == 'headline'].iterrows():
    print(f"{r['comparison']:<24} $r=${misc.write_scientific(r['r'])}, "
          f"{misc.write_perm_p(r['p'], n_nulls)} ({int(r['n_items'])} cells)")
print()
print('per mode, across the 24 norms:')
for _, r in norm_rep[norm_rep['block'] == 'per_mode'].iterrows():
    print(f"  {r['comparison']:<14} $r=${misc.write_scientific(r['r'])}, "
          f"{misc.write_perm_p(r['p'], n_nulls, note=False)}, "
          f"{misc.write_p(r['p_bh'], sub=fdr_sub)}")


# %%
# how much text each window actually got

cov_o = pd.read_csv(f'{orig_norm_dir}/text_coverage.csv').set_index('movie')
cov_r = pd.read_csv(f'{norm_dir}/text_coverage.csv').set_index('movie')
cov_s = pd.read_csv(f'{norm_dir}/caption_single/text_coverage.csv').set_index('movie')
print(f"tokens per window: original Back to the Future "
      f"{cov_o.loc['backtothefuture', 'mean_tokens']:.1f}, "
      f"replication span {cov_r.loc['backtothefuture', 'mean_tokens']:.1f}, "
      f"single {cov_s.loc['backtothefuture', 'mean_tokens']:.1f}")


# %%
# do above but by ranking


# %%
# Semantic-similarity network of modes
# (1 - residualized pooled-centroid cosine distance) between atoms
# in the movie-residualized embedding space
#
# thresholded on the pairwise permutation test, 
# but mind the direction: did that on distance not similarity
import networkx as nx

sem = np.load('nndb2_results/stimuli/embeddings/atom_semantics.npz')
pairs_df = pd.read_csv('nndb2_results/stimuli/embeddings/pairwise_atom_tests.csv')
atom_ids = sem['atom_ids'].astype(int)
cos_dist = sem['cos_res']  # residualized pooled-centroid cosine distance (diag 0)
with open(f'nndb2_results/dynamics_winner/{parcellation}/export/winner_states.pkl', 'rb') as f:
    legend = {int(k): str(v) for k, v in pickle.load(f)['legend'].items()}
names = [legend[a] for a in atom_ids]  # match order

edge_q = 0.05 # FDR cutoff on the pairwise separation test
drop_ns = False  # True drops non-significant edges instead of fading them
ns_alpha = 0.07 # how faint a non-significant edge goes
width_scale = 5.0

q_by_pair = {tuple(sorted((int(r.atom_a), int(r.atom_b)))): float(r.q_fdr)
             for r in pairs_df.itertuples()}

sim = 1.0 - cos_dist  # semantic similarity in [-1, 1]

G = nx.Graph()
G.add_nodes_from(names)
for i, j in zip(*np.triu_indices(len(names), k=1)):
    q = q_by_pair[tuple(sorted((int(atom_ids[i]), int(atom_ids[j]))))]
    if drop_ns and q >= edge_q:
        continue
    G.add_edge(names[i], names[j], weight=float(sim[i, j]), q=q)

edge_list = list(G.edges())
weights = np.array([G[u][v]['weight'] for u, v in edge_list])
qs = np.array([G[u][v]['q'] for u, v in edge_list])
sig = qs < edge_q

# scale between .3 and 1 for alpha values, then knock back whatever is n.s.
weights_alpha = 0.3 + 0.7 * (weights - weights.min()) / (weights.max() - weights.min())
weights_alpha = np.where(sig, weights_alpha, ns_alpha)
edge_widths = np.abs(weights) * width_scale
edge_widths = np.where(sig, edge_widths, edge_widths * 0.45)

#pos = nx.spring_layout(G, weight='weight', seed=38)
pos = {'Oneness': np.array([0.01766082, 0.77856377]),
 'Cooperation': np.array([0.60278986, 0.44469489]),
 'Preservation': np.array([0.53547004, 0.21986309]),
 'Migration': np.array([-0.04928734,  0.46769091]),
 'Copy': np.array([-0.38476133, -1.        ]),
 'Erasure': np.array([-0.5643615 , -0.83207967]),
 'Unification': np.array([-0.46694039, -0.46209666]),
 'Dissolution': np.array([-0.1887971 , -0.84018382]),
 'Encryption': np.array([0.12597508, 0.60807185]),
 'Decryption': np.array([0.26225186, 0.61547564])}

fig, ax = plt.subplots(figsize=(4, 2.6))
edges = nx.draw_networkx_edges(
    G, pos, ax=ax, edgelist=edge_list, width=edge_widths,
    edge_color=weights, edge_cmap=misc.div_cmap,
    edge_vmin=-1.0, edge_vmax=1.0,
    alpha=weights_alpha,
    connectionstyle="arc3,rad=0.7",
    arrows=True,
    arrowstyle="-")
nx.draw_networkx_nodes(
    G, pos, ax=ax,
    node_color=[misc.atoms_colors[n] for n in G.nodes()],
    node_size=250, edgecolors='black', linewidths=1.0)
ax.set_axis_off()
fig.tight_layout()
fig.savefig(f'{plots_dir}/atom_semantic_network.png', bbox_inches='tight', transparent=True)
plt.show()

print(f'{sig.sum()}/{len(sig)} pairs separate from the cir-shift null at q<{edge_q}')
ns = [f'{u}-{v}' for (u, v), s in zip(edge_list, sig) if not s]
print('cannot tell apart:', ', '.join(ns))


# %%
from matplotlib.colors import Normalize

fig, ax = plt.subplots(1, 1, figsize=(.8, .1))
cbar = plt.colorbar(cm.ScalarMappable(norm=Normalize(-1, 1), cmap=misc.div_cmap),
                    cax=ax, orientation='horizontal')
cbar.set_ticks([-1, 0, 1])
cbar.set_ticklabels(['-1', '0', '1'])
cbar.set_label('Semantic \n similarity', fontsize=rc_fontsize)
fig.savefig(f'{plots_dir}/atom_semantic_network_cbar.png', bbox_inches='tight', transparent=True)
plt.show()


# %%
# -oomnibus reality check that atom identity does indeed explain semantic var
# PERMANOVA-style test:
# SS_between of the atom centroids in movie-residualized PCA semantic space 
# vs a participant-wise circular-shift null. 
# pseudo-F
import json
from scipy.stats import gaussian_kde

sem = np.load('nndb2_results/stimuli/embeddings/atom_semantics.npz')
ssb_obs = float(sem['ssb_obs'])
ssb_null = sem['ssb_null']
F_obs = float(sem['F_obs'])
r2_obs = float(sem['r2_obs'])
n_perm = len(ssb_null)
n_atoms = len(sem['atom_ids'])
N_total = json.load(open('nndb2_results/stimuli/embeddings/summary.json'))['n_total_occurrences']

p_perm = (1 + int((ssb_null >= ssb_obs).sum())) / (n_perm + 1)
ss_total = ssb_obs / r2_obs  # R2 = SS_between / SS_total
F_null = (ssb_null / (n_atoms - 1)) / ((ss_total - ssb_null) / (N_total - n_atoms))

n_exceed = int((ssb_null >= ssb_obs).sum())
print(f'Omnibus atom x semantics (circular-shift null, n_perm={n_perm}):')
print(f'  pseudo-F = {F_obs:.3f}')
print(f'  R^2 = {r2_obs:.2e} ({r2_obs * 100:.3f}% of residual semantic variance)')
print(f'  p_perm = {p_perm:.2e} ({n_exceed}/{n_perm} null >= observed; permutation floor)')

fig, ax = plt.subplots(1, 1, figsize=(1.3 * 1.25, .8 * 1.25))
lo, hi = min(F_null.min(), F_obs), max(F_null.max(), F_obs)
pad = 0.1 * (hi - lo)
grid = np.linspace(lo - pad, hi + pad, 400)
dens = gaussian_kde(F_null)(grid)
ax.plot(grid, dens / dens.max(), color='0.6', label='Null')
ax.axvline(F_obs, color='crimson', lw=1.5, label='Observed')
ax.set_ylim(0, 1.05)
ax.set_xlabel('Semantic \n differentiation')
sns.despine(ax=ax)
ax.set_xticks([2, 4, 6]); ax.set_yticks([0, 1])
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, loc=(.95, 0.4), frameon=False,
          handlelength=1.0, handletextpad=0.4, labelspacing=0.1)
fig.savefig(f'{plots_dir}/atom_omnibus_null.png', bbox_inches='tight', transparent=True)
plt.show()

# sample, prevalence and the pairwise picture, all from the same run
esum = json.load(open('nndb2_results/stimuli/embeddings/summary.json'))
geo = esum['geometry']
prev = esum['atom_prevalence']
pairs_df = pd.read_csv('nndb2_results/stimuli/embeddings/pairwise_atom_tests.csv')

print()
print(f"sample: {esum['n_participants']} participants, all watching "
      f"{'one film' if esum['n_films'] == 1 else str(esum['n_films']) + ' films, one each'}, "
      f"HRF lag {abs(esum['offset'])} s")
print('prevalence: ' + ', '.join(f'{k} {v:.1%}' for k, v in
                                 sorted(prev.items(), key=lambda kv: -kv[1])))
print(f"atom identity tracks different semantic content, pseudo-$F$={geo['pseudo_F']:.2f}, "
      f"{misc.write_p(geo['p_perm'], floor=2e-4, sub='perm')}, but explains only "
      f"{geo['r2']:.3%} of residual semantic variance")
print(f"pairwise: {int(pairs_df['sig_q05'].sum())}/{len(pairs_df)} atom pairs separate after FDR")


# %%
# replicating the semantic geometry
o_sem = np.load(f'{orig_mvf_dir}/atom_semantics.npz')
r_sem = np.load(f'{mvf_dir}/atom_semantics.npz')
assert list(o_sem['atom_ids']) == list(r_sem['atom_ids']), 'atom order disagrees'

geo_csv = f'{mvf_dir}/replication_geometry.csv'
if os.path.exists(geo_csv) and not recompute:
    geo_rep = pd.read_csv(geo_csv)
else:
    rows = []
    # the pairwise picture is a symmetric matrix, so the null relabels its nodes
    res = misc.matrix_null_corr(o_sem['cos_res'], r_sem['cos_res'], n_nulls=n_nulls)
    rows.append({'target': 'Pairwise cosine distance', 'r': res['r'], 'p': res['p'],
                 'n_items': res['n_edges'], 'n_nulls': res['n_nulls']})
    for label, key in [('Centroid norm', 'norm_obs')]:
        res = misc.perm_corr(o_sem[key], r_sem[key], n_perm=n_nulls, labels=modes)
        rows.append({'target': label, 'r': res['r'], 'p': res['p'],
                     'n_items': res['n_items'], 'n_nulls': res['n_nulls']})
    geo_rep = pd.DataFrame(rows)
    geo_rep['p_bh'] = multi.multipletests(geo_rep['p'].values, method='fdr_bh')[1]
    geo_rep.to_csv(geo_csv, index=False)

for _, r in geo_rep.iterrows():
    print(f"{r['target']:<26} $r=${misc.write_scientific(r['r'])}, "
          f"{misc.write_perm_p(r['p'], int(r['n_nulls']))}, "
          f"{misc.write_p(r['p_bh'], sub=fdr_sub)}")

o_geo = json.load(open(f'{orig_mvf_dir}/summary.json'))
r_geo = json.load(open(f'{mvf_dir}/summary.json'))
print()
print('omnibus, original vs replication:')
for k, lab, fmt in [('pseudo_F', 'pseudo-F', '.3f'), ('r2', 'R2', '.2e'),
                    ('evr', 'PCA variance kept', '.1%')]:
    print(f"  {lab:<20} {o_geo['geometry'][k]:{fmt}} vs {r_geo['geometry'][k]:{fmt}}")
print(f"  {'pairs sig (FDR)':<20} {o_geo['geometry']['n_pairs_sig']}/45 vs "
      f"{r_geo['geometry']['n_pairs_sig']}/45")
print(f"  {'occurrences':<20} {o_geo['n_total_occurrences']:,} over {o_geo['n_films']} films vs "
      f"{r_geo['n_total_occurrences']:,} over {r_geo['n_films']}")

# asked claud bout this: R2 rises while pseudo-F falls is a contradiction: 
# pseudo-F is R2 with the sample size divided back out, and the replication carries a quarter of the occurrences.

# larger share of the semantic variance is atom-linked here, on less data!


# %%
mvf_dir = 'nndb2_results/stimuli/embeddings'
mvf_df = pd.read_csv(f'{mvf_dir}/feature_atom_zscores.csv')

axis_order = ['semantic_flux', 'speed_surprise', 'coherence_rise', 'context_asymmetry']
axis_label = {'semantic_flux': 'Flux', 'coherence_rise': 'Convergence',
              'context_asymmetry': 'Handover', 'speed_surprise': 'Rush'}

za = mvf_df.pivot(index='feature', columns='name', values='z')
qa = mvf_df.pivot(index='feature', columns='name', values='q_fdr')
# mode order is taken from the PUBLISHED run's flux ranking rather than re-derived from
# nndb2's own, so this heatmap can be read straight against the paper's. re-sorting on the
# replication's own z would reshuffle the rows and make the two figures uncomparable by eye
mode_ord = (pd.read_csv(f'{orig_mvf_dir}/feature_atom_zscores.csv')
            .pivot(index='feature', columns='name', values='z')
            .loc['semantic_flux'].sort_values(ascending=False).index.tolist())
za = za.loc[axis_order, mode_ord]
qa = qa.loc[axis_order, mode_ord]

from palettable.scientific.diverging import Vik_3 as dddmap
from palettable.cartocolors.diverging import ArmyRose_3 as cccc

dddmap = dddmap.mpl_colormap
dddmap = LinearSegmentedColormap.from_list('dddmap', dddmap(np.linspace(0.1, 0.9, 256)))

threshhh = 0.05


# %%
# move move make ya body groove
vmax = 6.0

# transposed: kinematic categories along the top, modes down the right
zz = za.T.values
qq = qa.T.values
n_row, n_col = zz.shape

fig, ax = plt.subplots(figsize=(2.4, 5.8))
im = ax.imshow(zz, cmap=dddmap, vmin=-vmax, vmax=vmax, aspect='equal')

for i in range(n_row):
    for j in range(n_col):
        if qq[i, j] < threshhh:
            ax.text(j, i + .2, misc.get_star(qq[i, j]), ha='center', va='center',
                    color='black', fontsize=rc_fontsize + 4, zorder=3)

ax.set_xticks(range(n_col))
ax.set_xticklabels([axis_label[f] for f in axis_order], rotation=35, ha='left',
                   rotation_mode='anchor', fontsize=rc_fontsize)
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

ax.set_yticks(range(n_row))
ax.set_yticklabels([misc.atoms_names[a] for a in mode_ord], fontsize=rc_fontsize)
ax.yaxis.set_ticks_position('right')
ax.yaxis.set_label_position('right')

for i in range(n_row + 1):
    ax.axhline(i - 0.5, color='black', lw=.9, zorder=1)
for j in range(n_col + 1):
    ax.axvline(j - 0.5, color='black', lw=.9, zorder=1)

cbar = fig.colorbar(im, ax=ax, location='left', aspect=7, pad=0.06, shrink=0.15,
                    anchor=(1, .9))
cbar.set_label(r'$z$', fontsize=rc_fontsize, labelpad=0)
fig.savefig(f'{plots_dir}/atom_movement_axes_heatmap.png', bbox_inches='tight', transparent=True)
plt.show()

omni = pd.read_csv(f'{mvf_dir}/feature_omnibus.csv')
print('Omnibus per axis (does kinemat featur vary acros modes)')
for _, r in omni.iterrows():
    print(f"  {axis_label[r['feature']]} $SS={r['SS_between']:.1f}$, "
          f"{misc.write_p(r['p_perm'], floor=2e-4, sub='perm')}, "
          f"{misc.write_p(r['q_fdr'], floor=2e-4, sub=fdr_sub)}")

print()
print('----------------')
print()

# how much of it survives, and how big it actually is
esig = mvf_df[mvf_df['q_fdr'] < threshhh]
eeff = esig['mean_sd_units'].abs()
print(f"{len(esig)}/{len(mvf_df)} mode x axis cells survive FDR, over "
      f"{esig['name'].nunique()}/10 modes")
print(f"effects are small: biggest {eeff.max():.3f} SD, median {eeff.median():.3f} SD")
print()

# every cell, for the text
for _, r in mvf_df.iterrows():
    print(f"  {axis_label[r['feature']]}: {misc.atoms_texcommands[r['name']]}; "
          f"$z={r['z']:+.2f}$, {r['mean_sd_units']:+.3f} SD, "
          f"{misc.write_p(r['p_perm'], floor=2e-4, sub='perm')}, "
          f"{misc.write_p(r['q_fdr'], floor=2e-4, sub=fdr_sub)}")


# %%
# replicating the kinematic axes
o_ax = z_table(f'{orig_mvf_dir}/feature_atom_zscores.csv')
r_ax = z_table(f'{mvf_dir}/feature_atom_zscores.csv')

ax_csv = f'{mvf_dir}/replication_axes.csv'
if os.path.exists(ax_csv) and not recompute:
    ax_rep = pd.read_csv(ax_csv)
else:
    res = misc.profile_null_corr(o_ax.loc[modes, axis_order].values,
                                 r_ax.loc[modes, axis_order].values, n_perm=n_nulls)
    head = [{'block': 'headline', 'comparison': 'Mode x axis table', 'r': res['r'],
             'p': res['p'], 'n_items': res['n_items']}]
    per = []
    for a in axis_order:
        res = misc.perm_corr(o_ax.loc[modes, a].values, r_ax.loc[modes, a].values,
                             n_perm=n_nulls, labels=modes)
        per.append({'block': 'per_axis', 'comparison': axis_label[a], 'r': res['r'],
                    'p': res['p'], 'n_items': res['n_items']})
    per = pd.DataFrame(per)
    per['p_bh'] = multi.multipletests(per['p'].values, method='fdr_bh')[1]
    ax_rep = pd.concat([pd.DataFrame(head), per], ignore_index=True)
    ax_rep.to_csv(ax_csv, index=False)

for _, r in ax_rep.iterrows():
    tail = '' if pd.isna(r.get('p_bh', np.nan)) else f", {misc.write_p(r['p_bh'], sub=fdr_sub)}"
    print(f"{r['comparison']:<20} $r=${misc.write_scientific(r['r'])}, "
          f"{misc.write_perm_p(r['p'], n_nulls, note=False)}{tail} "
          f"({int(r['n_items'])} items)")


# %%
# original against replication, every comparison on one set of axes
# panels a and b are the 240 atom x norm cells under the two caption modes, c is the
# 40 atom x axis cells. r in each corner is read off the saved csv, not recomputed here
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.7))

def _r(df, key):
    return float(df.loc[df['comparison'] == key, 'r'].iloc[0])

def _square(ax):
    lim = np.array([-1, 1]) * max(np.abs(ax.get_xlim()).max(), np.abs(ax.get_ylim()).max())
    ax.plot(lim, lim, color='0.6', lw=0.8, ls='--', zorder=1)
    ax.axhline(0, color='0.85', lw=0.6, zorder=0)
    ax.axvline(0, color='0.85', lw=0.6, zorder=0)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Original $z$')
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    sns.despine(ax=ax)

for ax, other, title, rr in [
        (axes[0], r_norm, 'Lexical norms, (span)', _r(norm_rep, 'Norm profiles (span)')),
        (axes[1], s_norm, 'Lexical norms, (single)', _r(norm_rep, 'Norm profiles (single)'))]:
    for m in modes:
        ax.scatter(o_norm.loc[m, norm_cols].values, other.loc[m, norm_cols].values,
                   s=40, color=misc.atoms_colors[m], edgecolor='black', linewidth=0.3,
                   label=misc.atoms_names[m], zorder=3)
    _square(ax)
    ax.set_title(title, fontsize=rc_fontsize, loc='left')
    ax.text(0.04, 0.95, f'$r$ = {rr:.2f}', transform=ax.transAxes, va='top',
            fontsize=rc_fontsize)

ax = axes[2]
axis_marker = dict(zip(axis_order, ['o', 's', '^', 'D']))
for a in axis_order:
    for m in modes:
        ax.scatter(o_ax.loc[m, a], r_ax.loc[m, a], s=40, marker=axis_marker[a],
                   color=misc.atoms_colors[m], edgecolor='black', linewidth=0.4, zorder=3)
_square(ax)
ax.set_title('Kinematic axes', fontsize=rc_fontsize, loc='left')
ax.text(0.04, 0.95, f"$r$ = {_r(ax_rep, 'Mode x axis table'):.2f}", transform=ax.transAxes,
        va='top', fontsize=rc_fontsize)
ax.legend(handles=[plt.Line2D([0], [0], marker=axis_marker[a], ls='', color='0.4',
                              markeredgecolor='black', label=axis_label[a])
                   for a in axis_order],
          loc='lower right', frameon=False, fontsize=rc_fontsize,
          bbox_to_anchor=(1.45, 0),
          handletextpad=0.2, labelspacing=0.2)

axes[0].set_ylabel('Replication $z$')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles[:len(modes)], labels[:len(modes)], loc='center left',
           bbox_to_anchor=(.97, 0.5), frameon=False, fontsize=rc_fontsize,
           handletextpad=0.2, labelspacing=0.3)

for lab, ax in zip('abc', axes):
    ax.text(-0.08, 1.12, lab, transform=ax.transAxes, fontsize=rc_fontsize,
            va='top', ha='right', fontweight='bold')

plt.tight_layout()
fig.savefig(f'{plots_dir}/replication_stimuli.png', bbox_inches='tight', transparent=True)
plt.show()


# %%

single_csv = f'{norm_dir}/replication_norms_single.csv'
if os.path.exists(single_csv) and not recompute:
    single_rep = pd.read_csv(single_csv)
else:
    rows = []
    for m in modes:
        res = misc.perm_corr(o_norm.loc[m, norm_cols].values,
                             s_norm.loc[m, norm_cols].values, n_perm=n_nulls)
        rows.append({'comparison': m, 'r': res['r'], 'p': res['p'], 'n_items': res['n_items']})
    single_rep = pd.DataFrame(rows)
    single_rep['p_bh'] = multi.multipletests(single_rep['p'].values, method='fdr_bh')[1]
    single_rep.to_csv(single_csv, index=False)

per_mode_r = {
    'norm_span': norm_rep[norm_rep['block'] == 'per_mode'].set_index('comparison'),
    'norm_single': single_rep.set_index('comparison'),
}


def mode_panels(kind, other, pooled_r, fname, supt):
    """One figure: pooled cloud, then the same cells one mode at a time."""
    fig, axes = plt.subplots(3, 4, figsize=(13.2, 9.8))
    axes = axes.ravel()

    if kind == 'axes':
        xs = {m: o_ax.loc[m, axis_order].values for m in modes}
        ys = {m: other.loc[m, axis_order].values for m in modes}
    else:
        xs = {m: o_norm.loc[m, norm_cols].values for m in modes}
        ys = {m: other.loc[m, norm_cols].values for m in modes}

    allx = np.concatenate([xs[m] for m in modes])
    ally = np.concatenate([ys[m] for m in modes])
    lim = np.array([-1, 1]) * max(np.abs(allx).max(), np.abs(ally).max()) * 1.08

    def frame(ax):
        ax.plot(lim, lim, color='0.6', lw=0.8, ls='--', zorder=1)
        ax.axhline(0, color='0.85', lw=0.6, zorder=0)
        ax.axvline(0, color='0.85', lw=0.6, zorder=0)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        sns.despine(ax=ax)

    ax = axes[0]
    for m in modes:
        if kind == 'axes':
            for k, a in enumerate(axis_order):
                ax.scatter(xs[m][k], ys[m][k], s=34, marker=axis_marker[a],
                           color=misc.atoms_colors[m], edgecolor='black',
                           linewidth=0.4, zorder=3)
        else:
            ax.scatter(xs[m], ys[m], s=16, color=misc.atoms_colors[m],
                       edgecolor='black', linewidth=0.3, zorder=3)
    frame(ax)
    ax.set_title('All modes', fontsize=rc_fontsize, loc='left')
    ax.text(0.05, 0.95, f'$r$ = {pooled_r:.2f}', transform=ax.transAxes, va='top',
            fontsize=rc_fontsize)

    for i, m in enumerate(modes, start=1):
        ax = axes[i]
        col = misc.atoms_colors[m]
        if kind == 'axes':
            for k, a in enumerate(axis_order):
                ax.scatter(xs[m][k], ys[m][k], s=50, marker=axis_marker[a], color=col,
                           edgecolor='black', linewidth=0.4, zorder=3)
        else:
            ax.scatter(xs[m], ys[m], s=22, color=col, edgecolor='black',
                       linewidth=0.3, zorder=3)
        frame(ax)
        ax.set_title(misc.atoms_names[m], fontsize=rc_fontsize, loc='left', color=col)
        if kind != 'axes':
            row = per_mode_r[kind].loc[m]
            # uncorrected, as the paper's other replication scatters are; the FDR
            # values are the ones quoted in the text. no mode changes side either way
            star = misc.get_star(row['p'])
            ax.text(0.05, 0.95, f"$r$ = {row['r']:.2f} {star}",
                    transform=ax.transAxes, va='top', fontsize=rc_fontsize)

    axes[-1].set_axis_off()
    if kind == 'axes':
        axes[-1].legend(handles=[plt.Line2D([0], [0], marker=axis_marker[a], ls='',
                                            color='0.4', markeredgecolor='black',
                                            label=axis_label[a]) for a in axis_order],
                        loc='center', frameon=False, fontsize=rc_fontsize,
                        handletextpad=0.3, labelspacing=0.4)

    for j in (0, 4, 8):
        axes[j].set_ylabel('Replication $z$')
    for j in (7, 8, 9, 10):
        axes[j].set_xlabel('Original $z$')
    #for lab, ax in zip('abcdefghijk', axes[:11]):
    #    ax.text(-0.12, 1.16, lab, transform=ax.transAxes, fontsize=rc_fontsize + 2,
    #            va='top', ha='right', fontweight='bold')

    fig.suptitle(supt, fontsize=rc_fontsize, x=0.02, ha='left')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(f'{plots_dir}/{fname}', bbox_inches='tight', transparent=True)
    plt.show()


mode_panels('norm_span', r_norm, _r(norm_rep, 'Norm profiles (span)'),
            'replication_stimuli_norm.png', '')
mode_panels('norm_single', s_norm, _r(norm_rep, 'Norm profiles (single)'),
            'replication_stimuli_single.png', '')
mode_panels('axes', r_ax, _r(ax_rep, 'Mode x axis table'),
            'replication_stimuli_axes.png', '')

print('per-mode agreement, span vs single captions (uncorrected, then FDR):')
for m in modes:
    a, b = per_mode_r['norm_span'].loc[m], per_mode_r['norm_single'].loc[m]
    print(f"  {m:<14} span $r=${a['r']:+.3f} ({misc.write_p(a['p'], sub='perm')}, "
          f"{misc.write_p(a['p_bh'], sub=fdr_sub)}), "
          f"single $r=${b['r']:+.3f} ({misc.write_p(b['p'], sub='perm')}, "
          f"{misc.write_p(b['p_bh'], sub=fdr_sub)})")


# %%
# check me picked measures actually cluster as anticipated ... prayers 4 george
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

fam_df = pd.read_csv(f'{mvf_dir}/family_members.csv')
cache = np.load(f'{mvf_dir}/measures_cache.npz')
bank_names = list(cache['names'])
cache_movies = sorted(k[2:] for k in cache.files if k.startswith('F_'))

cols = [bank_names.index(m) for m in fam_df['measure']]
X = np.vstack([cache[f'F_{m}'] for m in cache_movies])[:, cols] * fam_df['sign'].values
R = np.corrcoef(X, rowvar=False)
D = np.clip(1 - R, 0, 2)
np.fill_diagonal(D, 0)
link = linkage(squareform(D, checks=False), 'average')
leaves = dendrogram(link, no_plot=True)['leaves']

fam_cols = {'semantic_flux': "#920000", 'coherence_rise': "#AB5824",
            'context_asymmetry': "#176B56", 'speed_surprise': "#BA418A"}
fam_of = fam_df['family'].values

fig = plt.figure(figsize=(7.8, 7.8))
axd = fig.add_axes([0.18, 0.80, 0.705, 0.16])
dendrogram(link, ax=axd, no_labels=True, color_threshold=0, link_color_func=lambda k: '0.3')
axd.set_xlim(0, 10 * len(leaves))  # scipy puts leaf i at 10i+5; pin it against autoscale
axd.set_xticks([]); axd.set_yticks([])
for sp in axd.spines.values():
    sp.set_visible(False)

axs = fig.add_axes([0.18, 0.775, 0.705, 0.02])
for i, k in enumerate(leaves):
    axs.add_patch(plt.Rectangle((i, 0), 1, 1, color=fam_cols[fam_of[k]], lw=1, edgecolor='black'))
axs.set_xlim(0, len(leaves)); axs.set_ylim(0, 1)
axs.set_xticks([]); axs.set_yticks([])

axm = fig.add_axes([0.18, 0.06, 0.705, 0.705])
# aspect='equal' would re-centre the box inside its rect and break the alignment
im = axm.imshow(R[np.ix_(leaves, leaves)], cmap=cmr.iceburn, vmin=-1, vmax=1,
                aspect='auto')
axm.set_xticks([])
axm.set_yticks(range(len(leaves)))
axm.set_yticklabels([fam_df['measure'].values[k].replace('combined_', '').replace('_unfaded', '')
                     for k in leaves], fontsize=rc_fontsize - 4)
for i, k in enumerate(leaves):
    axm.get_yticklabels()[i].set_color(fam_cols[fam_of[k]])
axm.tick_params(length=0)

# add lines between each family
for f in np.unique(fam_of):
    idx = np.where(fam_of[leaves] == f)[0]
    if len(idx) > 1:
        axm.axhline(idx[-1] + 0.5, color=fam_cols[f], lw=1.5, zorder=3)
        axm.axvline(idx[-1] + 0.5, color=fam_cols[f], lw=1.5, zorder=3)

cax = fig.add_axes([0.92, 0.06, 0.018, 0.15])
cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r'$r$', fontsize=rc_fontsize)
handles = [plt.Line2D([0], [0], marker='s', ls='', color=c, label=axis_label[f])
           for f, c in fam_cols.items()]
axd.legend(handles=handles, loc='upper left', bbox_to_anchor=(.99, .8), frameon=False,
           fontsize=rc_fontsize - 4, handletextpad=0.3, labelspacing=0.25)
fig.savefig(f'{plots_dir}/kinematics_cluster.png', bbox_inches='tight')
plt.show()

# how cleanly does the label-free clustering recover the hand grouping?
lab = fcluster(link, 4, 'maxclust')
tab = pd.crosstab(pd.Series(fam_of, name='hand family'),
                  pd.Series(lab, name='data cluster'))
print(tab)
print(f'\n{int(tab.values.max(axis=1).sum())}/{len(fam_of)} cluster match baby!!!')


# %%
# check medoid measure corr

axis_corr = pd.read_csv(f'{mvf_dir}/axis_correlations.csv', index_col=0)
axis_corr = axis_corr.loc[axis_order, axis_order]

fig, ax = plt.subplots(figsize=(2, 2))
im = ax.imshow(axis_corr.values, cmap=cmr.prinsenvlag_r, vmin=-1, vmax=1)
for i in range(len(axis_order)):
    for j in range(len(axis_order)):
        ax.text(j, i, f'{axis_corr.values[i, j]:.2f}', ha='center', va='center',
                fontsize=rc_fontsize - 5,
                color='black' if abs(axis_corr.values[i, j]) < 0.7 else 'white')
ax.set_xticks(range(len(axis_order)))
ax.set_xticklabels([axis_label[a] for a in axis_order], rotation=90, fontsize=rc_fontsize - 3)
ax.set_yticks(range(len(axis_order)))
ax.set_yticklabels([axis_label[a] for a in axis_order], fontsize=rc_fontsize - 3)
ax.tick_params(length=0)
for i in range(len(axis_order) + 1):
    ax.axhline(i - 0.5, color='black', lw=.9)
    ax.axvline(i - 0.5, color='black', lw=.9)
#fig.savefig(f'{plots_dir}/atom_movement_axis_corr.png', bbox_inches='tight', transparent=True)
plt.show()

ev = np.linalg.eigvalsh(axis_corr.values)[::-1]
print(f'largest abso off-diagonal r = {np.abs(axis_corr.values - np.eye(4)).max():.3f}')
print(f'eigenvalss {np.round(ev, 3)}  ({int((ev > 1).sum())} above 1, '
      f'PC1 = {ev[0] / ev.sum():.1%} of variance)')


# %%
# supp fig: the full significance table behind the four kinematic axes
# every one of the 45 measures tested on its own against the circular-shift null,
# with r giving its agreement with the leave-one-out composite of its own family
# (the mean of the other members, so a measure is never checked against a mean it is in)

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from palettable.scientific.diverging import Vik_3
import misc

_font_dir = '/home/george/Research/fonts/Instrument_Sans/static'
_have = {(f.name, f.weight, f.style) for f in fm.fontManager.ttflist}
for _f in sorted(glob.glob(f'{_font_dir}/InstrumentSans-*.ttf')):
    try:
        _p = fm.FontProperties(fname=_f)
        if (_p.get_name(), _p.get_weight(), _p.get_style()) not in _have:
            fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams['font.family'] = 'Instrument Sans'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 800
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
rc_fontsize = 15
plt.rcParams.update({'font.size': rc_fontsize})

mvf_dir = 'nndb2_results/stimuli/embeddings'
orig_mvf_dir = '../results/stimuli/embeddings'
plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

axis_order = ['semantic_flux', 'speed_surprise', 'coherence_rise', 'context_asymmetry']
axis_label = {'semantic_flux': 'Flux', 'coherence_rise': 'Convergence',
              'context_asymmetry': 'Handover', 'speed_surprise': 'Rush'}
threshhh = 0.05
vmax = 6.0

mvf_df = pd.read_csv(f'{mvf_dir}/feature_atom_zscores.csv')
# mode order is taken from the PUBLISHED run's flux ranking rather than re-derived from
# nndb2's own, so this heatmap can be read straight against the paper's. re-sorting on the
# replication's own z would reshuffle the rows and make the two figures uncomparable by eye
mode_ord = (pd.read_csv(f'{orig_mvf_dir}/feature_atom_zscores.csv')
            .pivot(index='feature', columns='name', values='z')
            .loc['semantic_flux'].sort_values(ascending=False).index.tolist())

dddmap = LinearSegmentedColormap.from_list(
    'dddmap', Vik_3.mpl_colormap(np.linspace(0.1, 0.9, 256)))

fs_tick = rc_fontsize
fs_star = rc_fontsize
fs_lab = rc_fontsize

fam_df = pd.read_csv(f'{mvf_dir}/family_members.csv')
panel_r = pd.read_csv(f'{mvf_dir}/panel_profile_agreement.csv')
fam_cols = {'semantic_flux': "#920000", 'coherence_rise': "#AB5824",
            'context_asymmetry': "#176B56", 'speed_surprise': "#BA418A"}
tidy = lambda m: m.replace('combined_', '').replace('_unfaded', '')

# flux is 26 measures on its own, the three smaller families share the second column
columns = [['semantic_flux'], ['speed_surprise', 'coherence_rise', 'context_asymmetry']]

members_of = {f: fam_df.loc[fam_df['family'] == f, 'measure'].tolist() for f in axis_order}
rmap_of = {f: panel_r[panel_r['family'] == f].set_index('measure')['profile_r']
           for f in axis_order}
row_lab = {f: [f"{tidy(m)}  r={rmap_of[f][m]:+.2f}" for m in members_of[f]] for f in axis_order}

# explicit rects in inches, so every cell is square and identical across families
cell = 0.30
hm_w = len(mode_ord) * cell
char_w = 0.0068 * fs_tick
bar_pad, bar_w, name_w, col_gap = 0.10, 0.22, 0.55, 0.20
gap = 0.17
tick_h, pad_bot, pad_top, pad_left = 1.55, 0.25, 0.35, 0.05

# first pass: each column gets only as much label margin as its longest row name needs
hm_x, x_cursor = [], pad_left
for col in columns:
    lab_w = char_w * max(len(s) for f in col for s in row_lab[f]) + 0.12
    hm_x.append(x_cursor + lab_w)
    x_cursor += lab_w + hm_w + bar_pad + bar_w + name_w + col_gap
fig_w = x_cursor - col_gap + pad_left

col_h = [sum(len(members_of[f]) for f in col) * cell + gap * (len(col) - 1)
         for col in columns]
y_top_in = tick_h + pad_bot + max(col_h)
fig_h = y_top_in + pad_top

fig = plt.figure(figsize=(fig_w, fig_h))
rect = lambda x, y, w, h: [x / fig_w, y / fig_h, w / fig_w, h / fig_h]

for col, xh in zip(columns, hm_x):
    y_cur = y_top_in
    for fam in col:
        pdf = pd.read_csv(f'{mvf_dir}/panel_{fam}/feature_atom_zscores.csv')
        members = members_of[fam]
        pz = pdf.pivot(index='feature', columns='name', values='z').loc[members, mode_ord]
        pq = pdf.pivot(index='feature', columns='name', values='q_fdr').loc[members, mode_ord]
        h = len(members) * cell
        ax = fig.add_axes(rect(xh, y_cur - h, hm_w, h))
        im = ax.imshow(pz.values, cmap=dddmap, vmin=-vmax, vmax=vmax, aspect='auto')
        for i in range(pz.shape[0]):
            for j in range(pz.shape[1]):
                if pq.values[i, j] < threshhh:
                    ax.text(j, i + .2, misc.get_star(pq.values[i, j]), ha='center',
                            va='center', fontsize=fs_star)
        ax.set_yticks(range(len(members)))
        ax.set_yticklabels(row_lab[fam], fontsize=fs_tick)
        for t in ax.get_yticklabels():
            t.set_color(fam_cols[fam])
        ax.set_xticks(range(len(mode_ord)))
        ax.set_xticklabels([misc.atoms_names[a] for a in mode_ord] if fam == col[-1] else [],
                           rotation=90, fontsize=fs_tick)
        ax.tick_params(length=0)
        for i in range(len(members) + 1):
            ax.axhline(i - 0.5, color='black', lw=.4)
        for j in range(len(mode_ord) + 1):
            ax.axvline(j - 0.5, color='black', lw=.4)

        # category name on the right, against a colour bar spanning that family's rows
        axr = fig.add_axes(rect(xh + hm_w + bar_pad, y_cur - h, bar_w, h))
        axr.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=fam_cols[fam],
                                    edgecolor='black', lw=0.6))
        axr.set_xlim(0, 1); axr.set_ylim(0, 1)
        axr.set_xticks([]); axr.set_yticks([])
        for sp in axr.spines.values():
            sp.set_visible(False)
        axr.text(2.1, 0.5, axis_label[fam], transform=axr.transAxes, rotation=270,
                 va='center', ha='center', fontsize=fs_lab, color=fam_cols[fam])
        y_cur -= h + gap

short = int(np.argmin(col_h))
cax = fig.add_axes(rect(hm_x[short]+2, 2.5,
                        hm_w * 0.3, 0.12))
cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
cbar.set_label(r'$z$', fontsize=fs_lab)
cbar.ax.tick_params(labelsize=fs_tick)

fig.savefig(f'{plots_dir}/kinematics_all.png', bbox_inches='tight', transparent=True)
plt.show()

kinsum = json.load(open(f'{mvf_dir}/summary.json'))['kinematics']
for fam, r in kinsum['panel_profile_r'].items():
    print(f"  {axis_label[fam]:12s} {kinsum['families'][fam]['n_measures']:2d} measures, "
          f"leave-one-out agreement mean r={r['mean']:+.2f}, min {r['min']:+.2f}")


# %%
# the measure bank itself is imported from the root stimuli_utils rather than copied,
# so it cannot have drifted. this is the check that says so out loud, and it stands in
# for the supplementary LaTeX table, which the original run already generates and which
# would be identical here
o_fam = pd.read_csv(f'{orig_mvf_dir}/family_members.csv')
r_fam = pd.read_csv(f'{mvf_dir}/family_members.csv')

key = ['family', 'measure', 'sign', 'description']
assert o_fam[key].equals(r_fam[key]), 'the measure bank has drifted from the original run'
print(f'measure bank identical to the original run: {len(r_fam)} measures, '
      f'{r_fam["family"].nunique()} families, signs and descriptions all match')

# only r_to_family should move, since that is computed on nndb2's own trajectories
typ = o_fam[['family', 'measure']].copy()
typ['r_orig'] = o_fam['r_to_family']
typ['r_repl'] = r_fam['r_to_family']
res = misc.perm_corr(typ['r_orig'].values, typ['r_repl'].values, n_perm=n_nulls)
misc.report_replication('measure typicality', res)
for fam in axis_order:
    sub = typ[typ['family'] == fam]
    print(f"  {axis_label[fam]:<12} mean r_to_family {sub['r_orig'].mean():+.3f} vs "
          f"{sub['r_repl'].mean():+.3f} over {len(sub)} measures")
