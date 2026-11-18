# %%
# libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings
import cmasher as cmr
import palettable as pl
import pingouin as pg
from scipy.stats import spearmanr, zscore
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
dynamic_dir = f'results/main_dynamic/{parcellation}'
plots_dir = f'plots'
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

import sys
sys.path.append(f'{MRI_path}')
from cortexcanvas import plot_cortex

matching_results_dir = f'results/matching/{parcellation}'
if not os.path.exists(matching_results_dir):
    os.makedirs(matching_results_dir)

fmri_path = os.path.expanduser('~/Research/fMRI_general')
ahba_path = f'{fmri_path}/AHBA_genes'
neuros_path = f'{fmri_path}/neurosynth/concat/parcellated'


# %%
from palettable.cmocean.sequential import Dense_20_r
cmap = Dense_20_r.mpl_colormap
cmap = LinearSegmentedColormap.from_list('greyred', ["#939393", "#E21300"], N=1000)
cmap = LinearSegmentedColormap.from_list('grey_red_exp', cmap(np.linspace(0, 1, 1000)**3), N=1000)
example_receptors = ['GRIA1', 'GRIN1', 'HTR7', 'PVALB']

ahba_data = pd.read_pickle(f'{ahba_path}/{parcellation}_expression_dense.pkl')
ahba_rec_data = ahba_data.to_numpy()

for rec in example_receptors:
    idx = ahba_data.columns.get_loc(rec)
    data = ahba_rec_data[:, idx]
    fig = plot_cortex(data, parcellation, 
                      cmap=cmap, vmin=np.percentile(data, 5), vmax=np.percentile(data, 95),
                      views='lateral', hemi='rh')
    plt.savefig(f'{plots_dir}/cortex_{rec}.png', bbox_inches='tight', transparent=True)


# %%
neuros_path = f'{fmri_path}/neurosynth/concat/parcellated'
neuros_data = loadmat(f'{neuros_path}/{parcellation}_neurosynth.mat')
maps = neuros_data['data']
map_names = neuros_data['names']
map_names = [name.strip() for name in map_names]
map_names = [name.replace('_', '') for name in map_names]
maps = zscore(maps, axis=0) # zscore per map

example_maps = ['multisensory', 'inference', 'categorization', 'attention']

for map in example_maps:
    map_idx = map_names.index(map)
    map_data = maps[:, map_idx].flatten()
    fig = plot_cortex(map_data, parcellation, 
                      cmap=cmap, vmin=np.percentile(map_data, 5), vmax=np.percentile(map_data, 95),
                      views='lateral', hemi='rh')
    plt.savefig(f'{plots_dir}/cortex_{map}.png', bbox_inches='tight', transparent=True)


# %%
# plot time point for a specific subject for each mode of info

ex_timepoint = 2125

perp = misc.subjects[0]
mat_path = f'{dynamic_dir}/sub-{perp}/phiid_regional.mat'
mat = loadmat(mat_path)
for atom in misc.atoms_nr:
    data = mat[atom][:, ex_timepoint]
    print(f'{atom}')
    fig = plot_cortex(data, parcellation, 
                      cmap=misc.con_cmap, vmin=np.percentile(data, 5), vmax=np.percentile(data, 95),
                      views='lateral', hemi='rh')
    plt.savefig(f'{plots_dir}/timepoint_{atom}_sub-{perp}.png', bbox_inches='tight', transparent=True)


# %%
import datetime

def _excel_date_to_gene(x):
    # Excel silently coerces symbols like SEPT1/MARCH1/DEC1 into dates;
    # recover them (month -> gene family, day -> number) so an exact name match is not lost
    family = {3: 'MARCH', 9: 'SEPT', 12: 'DEC'}.get(x.month)
    return f'{family}{x.day}' if family else x

def get_burt_genes(sheet='Brain'):
    # Brain, Neuron, Synaptome, or Layers
    raw = pd.read_excel(f'{ahba_path}/Burt_NatNeuro2018.xlsx', sheet_name=sheet).to_numpy().flatten()
    return [_excel_date_to_gene(x) if isinstance(x, (pd.Timestamp, datetime.datetime)) else x for x in raw]

def _matching(data, maps, map_names, option='max'):

    regions, n_maps = maps.shape
    regions, timepoints = data.shape
    
    combined = np.vstack([maps.T, data.T])
    corr_matrix = np.corrcoef(combined)
    t_corrs = corr_matrix[:n_maps, n_maps:]
    
    if option == 'mean':
        t_corrs_agg = np.mean(t_corrs, axis=0)
    elif option == 'median':
        t_corrs_agg = np.median(t_corrs, axis=0)
    elif option == 'max':
        max_idxs = np.argmax(t_corrs, axis=0)
        max_names = [map_names[idx] for idx in max_idxs]
        t_corrs_agg = np.max(t_corrs, axis=0)
    else:
        raise ValueError('Option not recognised')
    
    # Average across timepoints
    m_index = np.mean(t_corrs_agg)
        
    return m_index, max_names


def molecular_matching(data, option='max', 
                       parcellation='extended_schaefer_1000', 
                       dtype=('genes', 'Brain'),
                       fmri_path=os.path.expanduser('~/Research/fMRI_general')):
    """
    Match topographies with molecular details
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data with shape (n_regions, n_timepoints)
    dtype : tuple, optional#
            Type of molecular data to match ('genes' or 'PET') and subtype (e.g. 'Brain' for genes)
    option : str, optional
        Method for aggregating correlations ('mean', 'median', or 'max')
    names : bool, optional
        Whether to return the names of the maximum correlations

    Returns:
    --------
    m_index : float
        Mean index across timepoints
    max_names : numpy.ndarray, optional
        Only returned if option='max', contains names of maximum correlations
    """

    if dtype[0] == 'genes':
    
        ahba_data = pd.read_pickle(f'{fmri_path}/AHBA_genes/{parcellation}_expression_dense.pkl')
        neurolist = get_burt_genes(dtype[1])

        ahba_rec_data = ahba_data[ahba_data.columns.intersection(neurolist)]
        ahba_rec_data = ahba_rec_data.reindex(columns=neurolist, fill_value=0)
        ahba_rec_data = ahba_rec_data.to_numpy()
        ahba_rec_data = np.nan_to_num(ahba_rec_data) # set nans to 0

        map_names = neurolist
        maps = ahba_rec_data

    if dtype[0] == 'PET':

        PET_path = f'{fmri_path}/PET_receptors/concat/parcellated'
        pet_data = loadmat(f'{PET_path}/{parcellation}_receptors.mat')

        map_names = pet_data['names']
        map_names = [name.strip() for name in map_names if name]
        maps = pet_data['data']
    
    return _matching(data, maps, map_names, option=option)

def cognitive_matching(data, option='max',
                       parcellation='extended_schaefer_1000',
                       cog_terms=True,
                       remove_regions=False,
                       fmri_path=os.path.expanduser('~/Research/fMRI_general')):
    """
    Match topographies with meta-analytic Neurosynth maps
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data with shape (n_regions, n_timepoints)
    cog_terms : bool, optional
        Whether to restrict to cognitive terms from Cognitive Atlas
    option : str, optional
        Method for aggregating correlations ('mean', 'median', or 'max')
    names : bool, optional
        Whether to return the names of the maximum correlations

    Returns:
    --------
    m_index : float
        Mean index across timepoints
    max_names : numpy.ndarray, optional
        Only returned if option='max', contains names of maximum correlations
    """
    
    neuros_path = f'{fmri_path}/neurosynth/concat/parcellated'
    neuros_data = loadmat(f'{neuros_path}/{parcellation}_neurosynth.mat')
    maps = neuros_data['data']
    map_names = neuros_data['names']
    map_names = [name.strip() for name in map_names]
    map_names = [name.replace('_', '') for name in map_names]

    if cog_terms:
        cog_terms_path = f'{fmri_path}/neurosynth/cognitive_terms_cogat.txt'
        with open(cog_terms_path, 'r') as f:
            cog_terms = f.readlines()
        cog_terms = [term.strip() for term in cog_terms]
        cog_indices = [i for i, name in enumerate(map_names) if name in cog_terms]
        maps = maps[:, cog_indices]
        map_names = [name for i, name in enumerate(map_names) if i in cog_indices]

    if remove_regions is not False:
        maps = np.delete(maps, remove_regions, axis=0)
        data = np.delete(data, remove_regions, axis=0)

    return _matching(data, maps, map_names, option=option)


# %%
gene_types = ['Brain', 'Neuron', 'Synaptome']
all_keys = gene_types + ['cog']
csv_paths = {key: f'{matching_results_dir}/matching_{key}.csv' for key in all_keys}

# always print gene availability
ahba_ = pd.read_pickle(f'{ahba_path}/{parcellation}_expression_dense.pkl')
for gtype in gene_types:
    neurolist = get_burt_genes(gtype)
    found = ahba_.columns.intersection(neurolist)
    missing = [g for g in neurolist if g not in found]
    print(f'{gtype}: {len(found)}/{len(neurolist)} genes found')
    if missing:
        print(f'  missing: {missing}')

if all(os.path.exists(p) for p in csv_paths.values()):
    dfs = {key: pd.read_csv(p, index_col='subject') for key, p in csv_paths.items()}
    print('Loaded from existing CSVs.')
else:
    def _zscore_rows(A):
        mu = A.mean(axis=1, keepdims=True)
        sig = A.std(axis=1, keepdims=True)
        sig[sig < 1e-12] = 1e-12
        return (A - mu) / sig

    # z-score maps once — avoids recomputing for every subject x atom call
    gene_maps_z = {}
    gene_map_names = {}
    for gtype in gene_types:
        neurolist = get_burt_genes(gtype)
        found = ahba_.columns.intersection(neurolist)
        arr = ahba_[found].to_numpy().astype(float)
        gene_map_names[gtype] = list(found)
        gene_maps_z[gtype] = _zscore_rows(arr.T)  # (n_genes, regions)

    neuros_raw = loadmat(f'{fmri_path}/neurosynth/concat/parcellated/{parcellation}_neurosynth.mat')
    cog_names_all = [n.strip().replace('_', '') for n in neuros_raw['names']]
    cog_maps_raw = neuros_raw['data'].astype(float)
    with open(f'{fmri_path}/neurosynth/cognitive_terms_cogat.txt') as f:
        cog_term_list = [l.strip() for l in f]
    cog_idx = [i for i, n in enumerate(cog_names_all) if n in cog_term_list]
    cog_maps_raw = cog_maps_raw[:, cog_idx]
    cog_names = [cog_names_all[i] for i in cog_idx]
    cog_maps_z = _zscore_rows(cog_maps_raw.T)  # (n_cog, regions)
    print(f'Cognitive: {len(cog_names)} terms, {np.any(np.isnan(cog_maps_raw), axis=1).sum()} NaN regions')

    n_reg = ahba_.shape[0]

    def _mean_max_corr(data_z, maps_z, n):
        # direct cross-correlation only — avoids the full (n_maps+timepoints)^2 corrcoef matrix
        # data_z: (timepoints, regions), maps_z: (n_maps, regions), both already z-scored
        t_corrs = (maps_z @ data_z.T) / n  # (n_maps, timepoints)
        return float(np.max(t_corrs, axis=0).mean())

    rows = {gtype: [] for gtype in gene_types}
    rows['cog'] = []

    for perp in tqdm(misc.subjects, desc='subjects'):
        mat_path = f'{dynamic_dir}/sub-{perp}/phiid_regional.mat'
        if not os.path.exists(mat_path):
            continue
        mat = loadmat(mat_path)
        gene_rows = {gtype: {'subject': perp} for gtype in gene_types}
        cog_row = {'subject': perp}

        for atom in misc.atom_order:
            data_raw = mat[atom].astype(np.float64)  # (regions, timepoints)
            nan_reg = np.any(np.isnan(data_raw), axis=1)
            if nan_reg.any():
                keep = ~nan_reg
                data_z = _zscore_rows(data_raw[keep].T)
                g_maps = {gtype: gene_maps_z[gtype][:, keep] for gtype in gene_types}
                c_maps = cog_maps_z[:, keep]
                n = keep.sum()
            else:
                data_z = _zscore_rows(data_raw.T)
                g_maps = gene_maps_z
                c_maps = cog_maps_z
                n = n_reg

            for gtype in gene_types:
                gene_rows[gtype][atom] = _mean_max_corr(data_z, g_maps[gtype], n)
            cog_row[atom] = _mean_max_corr(data_z, c_maps, n)

        for gtype in gene_types:
            rows[gtype].append(gene_rows[gtype])
        rows['cog'].append(cog_row)

    dfs = {}
    for key, rowlist in rows.items():
        df = pd.DataFrame(rowlist).set_index('subject')
        dfs[key] = df
        df.to_csv(csv_paths[key])
        print(f'{key}:\n{df.round(4)}\n')


# %%
import itertools

conditions = ['Brain', 'Neuron', 'Synaptome', 'cog']
cond_labels = {
    'Brain': 'Brain genes', 'Neuron': 'Neuron genes',
    'Synaptome': 'Synaptome genes', 'cog': 'Cognitive',
}

atom_pairs = list(itertools.combinations(misc.atom_order, 2))

for cond in conditions:
    df_cond = pd.read_csv(f'{matching_results_dir}/matching_{cond}.csv', index_col='subject')[misc.atom_order]
    df_long = df_cond.reset_index().melt(id_vars='subject', var_name='atom', value_name='value')

    anova = pg.rm_anova(data=df_long, dv='value', within='atom', subject='subject')

    pair_rows = []
    for atom_a, atom_b in atom_pairs:
        t, p, cil, ciu, d, bf, dof, power = misc.aug_t_test(
            df_cond[atom_a].values, df_cond[atom_b].values
        )
        pair_rows.append({'atom_a': atom_a, 'atom_b': atom_b, 't': t, 'p': p, 'd': d, 'bf': bf})
    pair_df = pd.DataFrame(pair_rows)
    _, pbh, _, _ = multi.multipletests(pair_df['p'].values, method='fdr_bh')
    pair_df['p_bh'] = pbh
    pair_df.to_csv(f'{matching_results_dir}/{cond}_pairwise.csv', index=False)

    F = misc.write_scientific(anova['F'].values[0])
    p_val = misc.write_scientific(anova['p_unc'].values[0])
    print(f"\n{cond_labels[cond]}: F({int(anova['ddof1'].values[0])},{int(anova['ddof2'].values[0])})={F}, p={p_val}")
    sig = pair_df[pair_df['p_bh'] < 0.05]
    print(f"  Significant pairs (p_FDR<0.05): {len(sig)}/{len(atom_pairs)}")
    for _, row in sig.iterrows():
        a = misc.atoms_names[row['atom_a']]; b = misc.atoms_names[row['atom_b']]
        t_s = misc.write_scientific(row['t']); p_s = misc.write_scientific(row['p_bh'])
        d_s = misc.write_scientific(row['d'])
        print(f"    {a} vs {b}: t={t_s}, p_FDR={p_s}, d={d_s}")

    # worst (largest p) contrast involving Cooperation
    coop_row = pair_df[(pair_df['atom_a'] == 'Cooperation') | (pair_df['atom_b'] == 'Cooperation')]
    if not coop_row.empty:
        row = coop_row.loc[coop_row['p_bh'].idxmax()]
        a = misc.atoms_names[row['atom_a']]; b = misc.atoms_names[row['atom_b']]
        t_s = misc.write_scientific(row['t']); p_s = misc.write_scientific(row['p_bh'])
        d_s = misc.write_scientific(row['d'])
        print(f"  Cooperation worst contrast: {a} vs {b}: t={t_s}, $p_{{\mathrm{{FDR}}}}$={p_s}, d={d_s}")


# %%
cond_labels = {
    'Brain': 'Brain genes', 'Neuron': 'Neuron genes',
    'Synaptome': 'Synapse genes', 'cog': 'Cognitive',
}
ylabels = {
    'Brain': 'Molecular match', 'Neuron': 'Molecular match',
    'Synaptome': 'Molecular match', 'cog': 'Cognitive match',
}

bp_kw = dict(
    showcaps=False, showfliers=False, patch_artist=True, widths=0.6,
    boxprops=dict(facecolor='none', edgecolor='black', linewidth=1, zorder=10),
    medianprops=dict(color='black', linewidth=1, zorder=11),
    whiskerprops=dict(color='black', linewidth=1, zorder=11),
)

def _fill_ax(ax, cond, seed=42):
    np.random.seed(seed)
    df_cond = pd.read_csv(f'{matching_results_dir}/matching_{cond}.csv', index_col='subject')
    for i, atom in enumerate(misc.atom_order):
        color = misc.atoms_colors[atom]
        vals = df_cond[atom].values
        ax.scatter(
            np.full(len(vals), i) + np.random.normal(0, 0.06, len(vals)),
            vals, color=color, alpha=0.6, s=12, zorder=5,
        )
        ax.boxplot(vals, positions=[i], **bp_kw)
    ax.set_xticks(range(len(misc.atom_order)))
    ax.set_xticklabels([misc.atoms_names[a] for a in misc.atom_order], rotation=90)
    for tick, atom in zip(ax.get_xticklabels(), misc.atom_order):
        tick.set_color(misc.atoms_colors[atom])
    ax.set_xlim(-0.5, len(misc.atom_order) - 0.5)
    ax.set_ylabel(ylabels[cond])
    if cond in ['Neuron', 'Synaptome']:
        ax.text(0.95, 0.95, cond_labels[cond], transform=ax.transAxes, 
                ha='right', va='top')
    ax.yaxis.set_major_locator(plt.MaxNLocator(2))
    sns.despine(ax=ax)

# Neuron + Synaptome side by side
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.25))
for ax, cond in zip(axes, ['Neuron', 'Synaptome']):
    _fill_ax(ax, cond)
plt.tight_layout()
fig.savefig(f'{plots_dir}/matching_Neuron_Synaptome.png', bbox_inches='tight')
plt.show()

# Brain
fig, ax = plt.subplots(figsize=(3.5, 3.5))
_fill_ax(ax, 'Brain')
plt.tight_layout()
fig.savefig(f'{plots_dir}/matching_Brain.png', bbox_inches='tight', transparent=True)
plt.show()

# Cognitive
fig, ax = plt.subplots(figsize=(3.5, 3.5))
_fill_ax(ax, 'cog')
plt.tight_layout()
fig.savefig(f'{plots_dir}/matching_cog.png', bbox_inches='tight', transparent=True)
plt.show()


# %%

gene_sheets = ['Brain', 'Neuron', 'Synaptome']
burt_lists = {s: set(get_burt_genes(s)) for s in gene_sheets}
ahba_genes = set(pd.read_pickle(f'{ahba_path}/{parcellation}_expression_dense.pkl').columns)

def check_gene(gene):
    in_ahba = gene in ahba_genes
    print(f'{gene}:  in AHBA expression = {in_ahba}')
    if not in_ahba:
        hit = [g for g in ahba_genes if g.upper() == gene.upper()]
        if hit:
            print(f'  (case mismatch? AHBA has {hit})')
    for s in gene_sheets:
        listed = gene in burt_lists[s]
        print(f'  {s:<10} listed={listed}  usable(list & AHBA)={listed and in_ahba}')

check_gene('PVALB')


# %%
import itertools

# chemo = AHBA gene matching (structure), cog = Neurosynth matching (function)
# >0.5 function-leaning, <0.5 structure-leaning, 0.5 balanced
chemo_key = 'Brain'
chemo_df = pd.read_csv(f'{matching_results_dir}/matching_{chemo_key}.csv', index_col='subject')[misc.atom_order]
cog_df = pd.read_csv(f'{matching_results_dir}/matching_cog.csv', index_col='subject')[misc.atom_order]
common = chemo_df.index.intersection(cog_df.index)
chemo_df, cog_df = chemo_df.loc[common], cog_df.loc[common]

sf_df = cog_df / (chemo_df + cog_df)
sf_df.to_csv(f'{matching_results_dir}/sf_ratio.csv')

# per mode: is the score reliably structure- or function-leaning (vs balanced 0.5)?
onesamp = []
for atom in misc.atom_order:
    res = pg.ttest(sf_df[atom].values, 0.5, alternative='two-sided')
    onesamp.append({'atom': atom, 't': float(res['T'].values[0]),
                    'p': float(res['p_val'].values[0]), 'd': float(res['cohen_d'].values[0])})
os_df = pd.DataFrame(onesamp)
_, os_df['p_bh'], _, _ = multi.multipletests(os_df['p'].values, method='fdr_bh')
os_df.to_csv(f'{matching_results_dir}/sf_ratio_onesample.csv', index=False)

print(f'Function-structure score  cog / ({chemo_key} + cog),  mean +/- sem,  one-sample vs 0.5:')
for _, r in os_df.iterrows():
    lean = 'function' if r['t'] > 0 else 'structure'
    m, s = sf_df[r['atom']].mean(), sf_df[r['atom']].sem()
    print(f"  {misc.atoms_names[r['atom']]:>18}  {m:.3f} +/- {s:.3f}   "
          f"t={misc.write_scientific(r['t'])}, p_FDR={misc.write_scientific(r['p_bh'])}, "
          f"d={misc.write_scientific(r['d'])}  ({lean}-leaning)")

# does the score differ across modes?
sf_long = sf_df.reset_index().melt(id_vars='subject', var_name='atom', value_name='value')
anova = pg.rm_anova(data=sf_long, dv='value', within='atom', subject='subject')
print(f"\nrm-ANOVA across modes: F({int(anova['ddof1'].values[0])},{int(anova['ddof2'].values[0])})="
      f"{misc.write_scientific(anova['F'].values[0])}, p={misc.write_scientific(anova['p_unc'].values[0])}")

pair_rows = []
for a, b in itertools.combinations(misc.atom_order, 2):
    t, p, cil, ciu, d, bf, dof, power = misc.aug_t_test(sf_df[a].values, sf_df[b].values)
    pair_rows.append({'atom_a': a, 'atom_b': b, 't': t, 'p': p, 'd': d})
pair_df = pd.DataFrame(pair_rows)
_, pair_df['p_bh'], _, _ = multi.multipletests(pair_df['p'].values, method='fdr_bh')
pair_df.to_csv(f'{matching_results_dir}/sf_ratio_pairwise.csv', index=False)
print(f"Significant pairwise differences (p_FDR<0.05): {int((pair_df['p_bh'] < 0.05).sum())}/{len(pair_df)}")

# worst (largest p) contrast involving Cooperation - even this is highly significant
coop_row = pair_df[(pair_df['atom_a'] == 'Cooperation') | (pair_df['atom_b'] == 'Cooperation')]
row = coop_row.loc[coop_row['p_bh'].idxmax()]
a, b = misc.atoms_names[row['atom_a']], misc.atoms_names[row['atom_b']]
print(f"Cooperation worst contrast: {a} vs {b}: t={misc.write_scientific(row['t'])}, "
      f"$p_{{\mathrm{{FDR}}}}$={misc.write_scientific(row['p_bh'])}, d={misc.write_scientific(row['d'])}")


# %%
# structure-function score per mode
np.random.seed(42)
fig, ax = plt.subplots(figsize=(3.5, 3))
sf_bp_kw = dict(
    showcaps=False, showfliers=False, patch_artist=True, widths=0.6,
    boxprops=dict(facecolor='none', edgecolor='black', linewidth=1, zorder=10),
    medianprops=dict(color='black', linewidth=1, zorder=11),
    whiskerprops=dict(color='black', linewidth=1, zorder=11),
)
for i, atom in enumerate(misc.atom_order):
    vals = sf_df[atom].values
    color = misc.atoms_colors[atom]
    ax.scatter(np.full(len(vals), i) + np.random.normal(0, 0.06, len(vals)),
               vals, color=color, alpha=0.6, s=12, zorder=5)
    ax.boxplot(vals, positions=[i], **sf_bp_kw)
#ax.axhline(0.5, color='grey', lw=1, ls='--', zorder=1)
ax.set_xticks(range(len(misc.atom_order)))
ax.set_xticklabels([misc.atoms_names[a] for a in misc.atom_order], rotation=90)
for tick, atom in zip(ax.get_xticklabels(), misc.atom_order):
    tick.set_color(misc.atoms_colors[atom])
ax.set_xlim(-0.5, len(misc.atom_order) - 0.5)
ax.set_ylabel('Functional bias')
ax.yaxis.set_major_locator(plt.MaxNLocator(2))
#ax.text(0.6, .9, 'function-like', transform=ax.transAxes, ha='left', va='top',
#        fontsize=10, color='lightgrey')
#ax.text(0.6, 0.0, 'structure-like', transform=ax.transAxes, ha='left', va='bottom',
#        fontsize=10, color='lightgrey')
sns.despine(ax=ax)
plt.tight_layout()
fig.savefig(f'{plots_dir}/sf_ratio.png', bbox_inches='tight', transparent=True)
plt.show()


# %%
# term appearance 

term_modalities = ['chemo', 'cog']
term_paths = {
    'chemo': f'{matching_results_dir}/term_freq_chemo_{chemo_key}.npz',
    'cog': f'{matching_results_dir}/term_freq_cog.npz',
}

if all(os.path.exists(p) for p in term_paths.values()):
    term_data = {m: np.load(p, allow_pickle=True) for m, p in term_paths.items()}
    print('Loaded term frequencies from cache.')
else:
    def _zscore_rows(a):
        mu = a.mean(axis=1, keepdims=True)
        sig = a.std(axis=1, keepdims=True)
        sig[sig < 1e-12] = 1e-12
        return (a - mu) / sig

    ahba_ = pd.read_pickle(f'{ahba_path}/{parcellation}_expression_dense.pkl')
    gene_names = list(ahba_.columns.intersection(get_burt_genes(chemo_key)))
    gene_maps_z = _zscore_rows(ahba_[gene_names].to_numpy().astype(np.float32).T)

    ns_raw = loadmat(f'{fmri_path}/neurosynth/concat/parcellated/{parcellation}_neurosynth.mat')
    cog_all = [n.strip().replace('_', '') for n in ns_raw['names']]
    with open(f'{fmri_path}/neurosynth/cognitive_terms_cogat.txt') as f:
        cog_list = [l.strip() for l in f]
    cog_sel = [i for i, n in enumerate(cog_all) if n in cog_list]
    cog_names = [cog_all[i] for i in cog_sel]
    cog_maps_z = _zscore_rows(ns_raw['data'][:, cog_sel].astype(np.float32).T)

    mod_maps = {'chemo': (gene_maps_z, gene_names), 'cog': (cog_maps_z, cog_names)}

    def _argmax_freq(data_raw, maps_z, n_terms):
        keep = ~np.any(np.isnan(data_raw), axis=1)
        data_z = _zscore_rows(data_raw[keep].T.astype(np.float32))
        corrs = maps_z[:, keep] @ data_z.T / keep.sum()
        return np.bincount(np.argmax(corrs, axis=0), minlength=n_terms) / corrs.shape[1]

    subj = [p for p in misc.subjects if os.path.exists(f'{dynamic_dir}/sub-{p}/phiid_regional.mat')]
    freqs = {m: np.zeros((len(misc.atom_order), len(subj), len(terms)), dtype=np.float32)
             for m, (mz, terms) in mod_maps.items()}
    for si, perp in enumerate(tqdm(subj, desc='subjects')):
        mat = loadmat(f'{dynamic_dir}/sub-{perp}/phiid_regional.mat')
        for ai, atom in enumerate(misc.atom_order):
            data_raw = mat[atom].astype(np.float64)
            for m, (maps_z, terms) in mod_maps.items():
                freqs[m][ai, si] = _argmax_freq(data_raw, maps_z, len(terms))

    term_data = {}
    for m, (maps_z, terms) in mod_maps.items():
        np.savez(term_paths[m], freq=freqs[m], atoms=np.array(misc.atom_order),
                 terms=np.array(terms), subjects=np.array(subj))
        term_data[m] = np.load(term_paths[m], allow_pickle=True)
    print('Computed and cached term frequencies.')

for m in term_modalities:
    d = term_data[m]
    print(f"{m}: freq {d['freq'].shape} = (atoms x subjects x terms), {len(d['terms'])} terms")


# %%
def plot_top_terms(modality, top_k=10, ncols=5):
    data = term_data[modality]
    freq = data['freq']  # (atoms, subjects, terms)
    terms = list(data['terms'])
    atoms = list(data['atoms'])
    n_subj = freq.shape[1]
    mean = freq.mean(axis=1)
    sem = freq.std(axis=1, ddof=1) / np.sqrt(n_subj)

    nrows = int(np.ceil(len(atoms) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.6, nrows * 2.7),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ai, atom in enumerate(atoms):
        ax = axes[ai]
        color = misc.atoms_colors[atom]
        order = np.argsort(mean[ai])[::-1][:top_k]
        y = np.arange(len(order))[::-1]
        ax.barh(y, mean[ai, order], xerr=sem[ai, order], color=color, alpha=0.85,
                height=0.72, error_kw=dict(ecolor='#333', lw=1, capsize=2), zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([terms[i] for i in order], fontsize=9)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_title(misc.atoms_names[atom], color=color, fontsize=13)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.tick_params(axis='x', labelsize=8)
        ax.margins(x=0.02)
        sns.despine(ax=ax)
    for j in range(len(atoms), len(axes)):
        axes[j].axis('off')
    
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(f'{plots_dir}/top_terms_{modality}.png', bbox_inches='tight', transparent=True)
    plt.show()

plot_top_terms('chemo')
plot_top_terms('cog')


# %%
# top recurring genes across modes, by mean best-match frequency
top_k = 20
for modality, width in [('chemo', 10), ('cog', 16)]:
    d = term_data[modality]
    names = np.array(d['terms'])
    overall = d['freq'].mean(axis=1).mean(axis=0)
    order = np.argsort(overall)[::-1][:top_k]
    label = f'genes ({chemo_key})' if modality == 'chemo' else 'cognitive terms'
    print(f'\nTop {top_k} recurring {label} across modes:')
    for rank, i in enumerate(order, 1):
        print(f'  {rank}. {names[i]} mean freq = {overall[i]:.4f}')
