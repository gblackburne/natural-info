import os
from functools import lru_cache
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import mannwhitneyu, rankdata
from scipy import signal
import cmasher as cmr

parcellation = 'extended_schaefer_1000'
atlas_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '../..', 'fMRI_general', 'atlases', 'schaefer_extend_atlases')
n_regions = 1054
subjects = [
    '04', '06', '07', '08', '09', '11', '12', '13',
    '14', '16', '17', '18', '19', '20', '21', '22',
    '23', '25', '26', '27', '29', '30', '31', '32',
    '33', '35', '37', '38', '39', '40', '42', '43',
    '44'
]
regions = np.arange(n_regions)
TR = 1.5
sfreq = 1/(1.5)
tau = 1

phiid_atoms = list(np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        '..', 'IG_phiid', 'phiid_keys.npy'),
                           allow_pickle=True))

atoms = {
    'Oneness': [0],
    'Cooperation': [15],
    'Preservation': [5, 10],
    'Erasure': [1, 2],
    'Copy':[4, 8],
    'Migration': [6, 9],
    'Unification':[7, 11],
    'Dissolution':[13, 14],
    'Decryption': [12],
    'Encryption': [3],
}
group_atoms = {
    'Oneness': ['rtr'],
    'Cooperation': ['sts'],
    'Preservation': ['xta', 'ytb'],
    'Erasure': ['rta', 'rtb'],
    'Copy': ['xtr', 'ytr'],
    'Migration': ['xtb', 'yta'],
    'Unification': ['xts', 'yts'],
    'Dissolution': ['sta', 'stb'],
    'Decryption': ['str'],
    'Encryption': ['rts'],
}

assert all(phiid_atoms[idx] == code
           for name, indices in atoms.items()
           for idx, code in zip(indices, group_atoms[name])), \
    "phiid_atoms and atoms dict are out of sync!"

atoms_names = {
    'Oneness': r'$R\!\rightarrow\!R$',
    'Cooperation': r'$S\!\rightarrow\!S$',
    'Preservation': r'$U_X\!\rightarrow\!U_X$',
    'Erasure': r'$R\!\rightarrow\!U$',
    'Copy': r'$U\!\rightarrow\!R$',
    'Migration': r'$U_X\!\rightarrow\!U_Y$',
    'Unification': r'$U\!\rightarrow\!S$',
    'Dissolution': r'$S\!\rightarrow\!U$',
    'Decryption': r'$S\!\rightarrow\!R$',
    'Encryption': r'$R\!\rightarrow\!S$',
}

atoms_texcommands = {
    'Oneness': r'\rtr',
    'Cooperation': r'\sts',
    'Preservation': r'\xtx',
    'Erasure': r'\rtx',
    'Copy': r'\xtr',
    'Migration': r'\xty',
    'Unification': r'\xts',
    'Dissolution': r'\stx',
    'Decryption': r'\str',
    'Encryption': r'\rts',
}

atoms_colors = {
    'Oneness': '#185948',
    'Cooperation': '#6B1E4E',
    'Preservation': '#F7965A',
    'Migration': '#F69289',
    'Copy': '#32A1C2',
    'Erasure': '#A2D6E4',
    'Unification': '#5C4596',
    'Dissolution': '#B1A5D4',
    'Encryption': '#827717',
    'Decryption': '#C5BE8B',
    'Migration': '#F69289',
}

atom_order = [
    'Oneness',
    'Cooperation',
    'Preservation',
    'Migration',
    'Copy',
    'Erasure',
    'Unification',
    'Dissolution',
    'Encryption',
    'Decryption',
]

atoms_nr = ['Oneness', 
           'Cooperation', 
           'Preservation', 
           'Migration', 
           'Copy', 
           'Unification', 
           'Encryption']

atoms_pol = ['Oneness',
           'Cooperation',
           'Preservation']

def truncate_colormap(cmap, minval=0.1, maxval=1.0, n=100):
    from matplotlib.colors import LinearSegmentedColormap
    new_cmap = LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

con_cmap = truncate_colormap(cmr.ghostlight, 0.1, 1.0)
div_cmap = truncate_colormap(cmr.pride, 0.1, 0.9)

from nilearn.plotting import cm as nlcm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, rgb_to_hsv, hsv_to_rgb
hcp_cmap = rgb_to_hsv(nlcm._cmap_d['videen_style'](np.linspace(0.5, 1.0, 1000))[:, :3])
hcp_cmap[:, 1] *= 0.85
hcp_cmap = ListedColormap(hsv_to_rgb(hcp_cmap))

brain_grey = "#8E8D8D" 
colors_full = [
    "#67b78e","#1D5B4A", brain_grey, "#4f062d","#f3a0d1"]
nodes = [0.0, 0.25, 0.5, 0.75, 1.0]
sr_full_cmap = LinearSegmentedColormap.from_list("sr_diverging_full", 
                                                 list(zip(nodes, colors_full)))

def write_scientific(val):
    val = float(val)
    # make it x 10^y
    if abs(val) < 0.001 or abs(val) > 1000:
        val = f'{val:.2e}'
        val = val.split('e')
        x= f'{val[0]}$\\times10^{{{int(val[1])}}}$'
    else:
        x= f'{val:.3f}'
    return x

def write_p(p, floor=1e-300, sub=None):
    # report a p-value, or an upper bound if it has underflowed below floor
    p = float(p)
    lab = f'$p_{{{sub}}}$' if sub else 'p'
    if p < floor:
        exp = int(np.floor(np.log10(floor)))
        if sub:
            return f'$p_{{{sub}}}<10^{{{exp}}}$'
        return f'p$<10^{{{exp}}}$'
    return f'{lab}={write_scientific(p)}'

def write_perm_p(p, n_nulls, sub='perm', note=True):
    """
    Report a permutation p-value, as an upper bound once it sits on its floor.

    A permutation p cannot fall below 1/(n_nulls + 1). That value does not mean
    "p equals 0.000999", it means no surrogate out of n_nulls matched or exceeded
    the observation, which bounds the tail rather than estimating it. Printing the
    number invites it to be read as a measurement, so at the floor this returns
    p < 1/n_nulls instead, and says so in words unless note is False.
    """
    p = float(p)
    floor = 1.0 / (n_nulls + 1)
    if p <= floor + 1e-12:
        exp = int(np.floor(np.log10(1.0 / n_nulls)))
        out = f'$p_{{{sub}}}<10^{{{exp}}}$'
        return out + f' [permutation floor, {n_nulls} nulls]' if note else out
    return f'$p_{{{sub}}}=${write_scientific(p)}'

def get_star(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s'

def aug_t_test(array1, array2, paired=True):
    '''
    T-test

    Parameters
    ----------
    array1: array
    array2: array

    array1 first so positive T mean array1 > array2

    '''

    result = pg.ttest(array1, array2, paired=paired, alternative='two-sided')

    t = float(result['T'].values[0])
    p = float(result['p_val'].values[0])
    cil = float(result['CI95'].values[0][0])
    ciu = float(result['CI95'].values[0][1])
    d = float(result['cohen_d'].values[0])
    bf = float(result['BF10'].values[0])
    dof = float(result['dof'].values[0])
    power = float(result['power'].values[0])

    return t, p, cil, ciu, d, bf, dof, power

def aug_mwu_test(x, y):
    '''
    Augmented Mann-Whitney U test.
    Positive Z and rbc indicate x > y stochastically.
    '''

    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    x = x[~np.isnan(x)]  # Manual handle for 'omit' logic
    y = y[~np.isnan(y)]

    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, np.nan, np.nan

    # SciPy 1.7+ returns U for the first sample (x)
    U, p = mannwhitneyu(x, y, alternative='two-sided')

    # Rank-Biserial Correlation (Effect Size)
    # etween -1 and 1
    rbc = (2 * U / (n1 * n2)) - 1

    # Z-score with Tie Correction
    combined = np.concatenate([x, y])
    n = n1 + n2
    _, t = np.unique(rankdata(combined), return_counts=True)
    mu = (n1 * n2) / 2
    sigma = np.sqrt((n1 * n2 / 12) * ((n + 1) - (np.sum(t**3 - t) / (n * (n - 1)))))

    if U > mu:
        Z = (U - mu - 0.5) / sigma
    elif U < mu:
        Z = (U - mu + 0.5) / sigma
    else:
        Z = 0.0

    return Z, p, rbc, U

def bandpass_filter(data, sfreq, low_cut=0.008, high_cut=0.09, filter_order=2):
    """Zero-phase Butterworth bandpass filter. data: (n_regions, n_samples)."""
    nyquist = 0.5 * sfreq
    sos = signal.butter(filter_order, [low_cut / nyquist, high_cut / nyquist],
                        btype='band', output='sos')
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):
        filtered[i] = signal.sosfiltfilt(sos, data[i])
    return filtered


def distinguishable_colors(n_colors, bg=None, saturation_range=(0, 1), lightness_range=(0, 1), as_hex=False,
                           sort=True):

    from skimage import color as skcolor
    from matplotlib.colors import rgb_to_hsv

    if bg is None:
        bg = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64)
    else:
        bg = np.array(bg, dtype=np.float64)

    n_grid = 30
    grid = np.linspace(0, 1, n_grid)
    r, g, b = np.meshgrid(grid, grid, grid)
    candidates_rgb = np.column_stack([r.ravel(), g.ravel(), b.ravel()])

    hsv_all = rgb_to_hsv(candidates_rgb)
    mask = (
        (hsv_all[:, 1] >= saturation_range[0]) & (hsv_all[:, 1] <= saturation_range[1]) &
        (hsv_all[:, 2] >= lightness_range[0])  & (hsv_all[:, 2] <= lightness_range[1])
    )
    candidates_rgb = candidates_rgb[mask]

    def to_lab(rgb):
        return skcolor.rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3)

    candidates_lab = to_lab(candidates_rgb)
    chosen_lab = to_lab(bg).tolist()
    chosen_rgb = []

    for _ in range(n_colors):
        chosen_arr = np.array(chosen_lab)
        diff = candidates_lab[:, None, :] - chosen_arr[None, :, :]
        min_sq_dists = (diff ** 2).sum(axis=2).min(axis=1)
        best = int(np.argmax(min_sq_dists))
        chosen_rgb.append(candidates_rgb[best])
        chosen_lab.append(candidates_lab[best])

    colors = np.array(chosen_rgb)
    lab = to_lab(colors)
    hsv = rgb_to_hsv(colors)
    if sort:
        order = np.argsort(hsv[:, 0] + 0.001 * (lab[:, 0] / 100.0))
        colors = colors[order]
    if as_hex:
        return ['#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in colors]
    return colors


@lru_cache(maxsize=None)
def load_atlas(parc=parcellation):
    df = pd.read_csv(os.path.join(atlas_dir, f'{parc}.csv'))
    abbr = pd.read_csv(os.path.join(atlas_dir, 'extended_schaefer_abbreviations.csv'))
    abbr_map = dict(zip(abbr['Abbreviation'], abbr['Full']))
    return df, abbr_map

def expand_region_name(region, abbr_map):
    for cand in (region, region.replace(' ', '-')):
        if cand in abbr_map:
            return abbr_map[cand]
    tokens = region.split()
    idx = ''
    if tokens and tokens[-1].isdigit():
        idx = ' ' + tokens[-1]
        tokens = tokens[:-1]
    stem = ' '.join(tokens)
    for cand in (stem, stem.replace(' ', '-'), tokens[0] if tokens else ''):
        if cand and cand in abbr_map:
            return abbr_map[cand] + idx
    return region

def print_topography_extremes(values, n=10, parc=parcellation, expand=True):
    df, abbr_map = load_atlas(parc)
    values = np.asarray(values, dtype=float).flatten()
    if len(values) > len(df):
        raise ValueError(f'topography has {len(values)} values but atlas has {len(df)} regions')

    out = df.iloc[:len(values)].copy()
    out['value'] = values
    out['name'] = out['region'].map(lambda r: expand_region_name(r, abbr_map)) if expand else out['region']
    out = out.dropna(subset=['value']).sort_values('value', ascending=False)

    top = out.head(n)
    bottom = out.tail(n).iloc[::-1]

    name_w = max(out['name'].str.len().max(), 6)
    for title, block in (('Top', top), ('Bottom', bottom)):
        print(f'{title} {len(block)} regions')
        print(f'  {"value":>9}  {"hemi":<4} {"name":<{name_w}} region')
        for _, row in block.iterrows():
            print(f'  {row["value"]:>9.4f}  {row["hemisphere"]:<4} {row["name"]:<{name_w}} ({row["region"]})')
        print()

    return top, bottom

def print_edge_extremes(matrix, n=10, parc=parcellation, expand=True):
    df, abbr_map = load_atlas(parc)
    matrix = np.asarray(matrix, dtype=float)
    r = matrix.shape[0]
    if r > len(df):
        raise ValueError(f'matrix has {r} regions but atlas has {len(df)} regions')

    reg = df['region'].iloc[:r]
    names = reg.map(lambda x: expand_region_name(x, abbr_map)).values if expand else reg.values
    hemis = df['hemisphere'].iloc[:r].values
    labels = [f'{h} {nm}' for h, nm in zip(hemis, names)]

    iu = np.triu_indices(r, k=1)
    vals = matrix[iu]
    keep = ~np.isnan(vals)
    ai, bi, vals = iu[0][keep], iu[1][keep], vals[keep]

    order = np.argsort(vals)[::-1]
    label_w = max(len(l) for l in labels)

    for title, sel in (('Top', order[:n]), ('Bottom', order[-n:][::-1])):
        print(f'{title} {len(sel)} edges')
        print(f'  {"value":>9}  {"region a":<{label_w}}      region b')
        for k in sel:
            print(f'  {vals[k]:>9.4f}  {labels[ai[k]]:<{label_w}}  <->  {labels[bi[k]]}')
        print()

    return ai[order[:n]], bi[order[:n]], vals[order[:n]]


# --- replication statistics -------------------------------------------------
#
# helpers for quantifying whether a result from the original dataset reappears
# in nndb2. three flavours, matched to the three shapes results come in:
#   regional maps  -> spatial_null_corr, variogram-matched surrogates (BrainSMASH)
#   connectomes    -> matrix_null_corr, node-relabelling null
#   per-mode vectors -> perm_corr, exhaustive/random label permutation over modes

@lru_cache(maxsize=None)
def region_distance_matrix(parc=parcellation):
    """Euclidean distance between parcel centroids, (n_regions, n_regions)."""
    from scipy.spatial import distance
    df, _ = load_atlas(parc)
    coords = df[['x', 'y', 'z']].values.astype(float)
    return distance.squareform(distance.pdist(coords))


def _corr(x, y, method):
    from scipy.stats import pearsonr, spearmanr
    if method == 'spearman':
        return float(spearmanr(x, y).statistic)
    if method == 'pearson':
        return float(pearsonr(x, y).statistic)
    raise ValueError(f'unknown method {method!r}')


def spatial_null_corr(x, y, dist=None, n_nulls=1000, seed=0, method='spearman',
                      parc=parcellation, resample=True, n_jobs=16, deltas=None,
                      **base_kwargs):
    """
    Correlate two regional maps, tested against spatially autocorrelated surrogates.

    Surrogates of ``x`` are generated with BrainSMASH's variogram-matching
    (Burt et al. 2020), which preserves the empirical distance-dependence of the
    map, so the null is "a map as spatially smooth as this one" rather than
    "any reshuffling of these values". ``resample=True`` additionally forces each
    surrogate to carry the same values as ``x``, only rearranged.

    Parameters
    ----------
    x, y : array_like, (n_regions,)
        Regional maps. Regions that are nan in either map are dropped from both.
    dist : array_like, optional
        (n_regions, n_regions) distance matrix. Defaults to Euclidean distance
        between parcel centroids of ``parc``.
    n_nulls : int
    method : {'spearman', 'pearson'}
    n_jobs : int
        Surrogate generation parallelises cleanly; 16 matches the local box.
    deltas : array_like, optional
        Smoothing levels swept when matching the variogram. Defaults to five levels
        spanning 0.1 to 0.9.

    Returns
    -------
    dict with keys r, p, n_nulls, n_regions, null (the n_nulls surrogate rs).
    ``p`` is two-tailed: the fraction of surrogates whose |r| meets or exceeds
    the observed |r|, with the customary +1 in numerator and denominator.
    """
    from brainsmash.mapgen.base import Base

    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    if x.shape != y.shape:
        raise ValueError(f'maps have different lengths: {x.shape} vs {y.shape}')

    dist = region_distance_matrix(parc) if dist is None else np.asarray(dist, dtype=float)
    if dist.shape[0] < len(x):
        raise ValueError(f'distance matrix covers {dist.shape[0]} regions, maps have {len(x)}')
    dist = dist[:len(x), :len(x)]

    keep = np.isfinite(x) & np.isfinite(y)
    x, y, dist = x[keep], y[keep], dist[np.ix_(keep, keep)]

    r_obs = _corr(x, y, method)

    # five smoothing levels rather than BrainSMASH's nine. Each surrogate is built
    # once per delta and the best-fitting kept, so the grid sets the runtime: nine
    # levels cost 13.3 s per 100 surrogates against 6.8 s for five, for a null
    # distribution of practically the same width (SD 0.163 vs 0.153). Three levels
    # would be faster again but the surrogates stop matching the variogram, and the
    # null widens to 0.214
    if deltas is None:
        deltas = np.arange(0.1, 1.0, 0.2)

    gen = Base(x=x, D=dist, resample=resample, seed=seed,
               n_jobs=n_jobs, deltas=deltas, **base_kwargs)
    surrogates = gen(n=n_nulls)
    null = np.array([_corr(s, y, method) for s in surrogates])

    p = (np.sum(np.abs(null) >= abs(r_obs)) + 1) / (n_nulls + 1)
    return {'r': r_obs, 'p': float(p), 'n_nulls': n_nulls,
            'n_regions': int(keep.sum()), 'null': null}


def matrix_null_corr(a, b, n_nulls=1000, seed=0, method='spearman', directed=False):
    """
    Edge-wise correlation between two matrices, against a node-relabelling null.
    Permuting the rows and columns of ``a`` together preserves its degree
    sequence and its edge-weight distribution while destroying the correspondence
    of its topography to ``b``.

    Parameters
    ----------
    a, b : array_like, (n, n)
    directed : bool
        ``False`` (default) reads the upper triangle only, for symmetric matrices
        such as connectomes or edge-wise ICC. ``True`` reads every off-diagonal
        cell, keeping ``a[i, j]`` and ``a[j, i]`` as separate observations, which
        is what a directed quantity like the TINDA FO asymmetry needs: the whole
        content of that matrix is its antisymmetry, and the upper triangle alone
        would discard half of it.

    Returns
    -------
    dict shaped like :func:`spatial_null_corr`, with ``n_edges`` in place of
    ``n_regions``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f'matrices have different shapes: {a.shape} vs {b.shape}')

    if directed:
        sel = ~np.eye(a.shape[0], dtype=bool)
    else:
        sel = np.zeros(a.shape, dtype=bool)
        sel[np.triu_indices(a.shape[0], k=1)] = True

    keep = np.isfinite(a[sel]) & np.isfinite(b[sel])
    b_vec = b[sel][keep]
    r_obs = _corr(a[sel][keep], b_vec, method)

    rng = np.random.default_rng(seed)
    null = np.empty(n_nulls)
    for i in range(n_nulls):
        perm = rng.permutation(a.shape[0])
        null[i] = _corr(a[np.ix_(perm, perm)][sel][keep], b_vec, method)

    p = (np.sum(np.abs(null) >= abs(r_obs)) + 1) / (n_nulls + 1)
    return {'r': r_obs, 'p': float(p), 'n_nulls': n_nulls,
            'n_edges': int(keep.sum()), 'null': null}


def perm_corr(x, y, n_perm=10000, seed=0, method='spearman', labels=None):
    """
    Correlation between two short vectors indexed by the same labels (typically the
    ten modes), tested by permuting one vector's labels. With ten modes the
    permutation distribution has 10! members, so ``n_perm`` random draws are used.

    ``labels`` is optional and only carried through to the result for reporting.
    """
    x = np.asarray(x, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()
    if x.shape != y.shape:
        raise ValueError(f'vectors have different lengths: {x.shape} vs {y.shape}')

    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    r_obs = _corr(x, y, method)

    rng = np.random.default_rng(seed)
    null = np.array([_corr(rng.permutation(x), y, method) for _ in range(n_perm)])

    p = (np.sum(np.abs(null) >= abs(r_obs)) + 1) / (n_perm + 1)
    return {'r': r_obs, 'p': float(p), 'n_nulls': n_perm, 'n_items': int(keep.sum()),
            'null': null, 'labels': labels}


def report_replication(label, res, width=28):
    """One-line summary of a null-model result, in the notation used in the paper."""
    n = res.get('n_regions', res.get('n_edges', res.get('n_items')))
    unit = ('regions' if 'n_regions' in res
            else 'edges' if 'n_edges' in res else 'items')
    print(f'{label:<{width}} r = {res["r"]:+.3f}, {write_p(res["p"], sub="null")}  '
          f'({n} {unit}, {res["n_nulls"]} nulls)')
    return res


def profile_null_corr(a, b, n_perm=10000, seed=0, method='spearman', labels=None):
    """
    Correlation between two mode x feature tables, against a mode-relabelling null.

    ``perm_corr`` permutes a single vector, which is the wrong null for a table: with
    240 cells, freely shuffling them tests against noise rather than against the thing
    that could actually go wrong, namely the modes carrying the wrong profiles. Here
    the rows of ``a`` are permuted as blocks, so each surrogate keeps every profile
    intact and only reassigns which mode owns it. With ten modes the permutation
    distribution has 10! members, so ``n_perm`` random draws are used.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f'tables have different shapes: {a.shape} vs {b.shape}')

    keep = np.isfinite(a) & np.isfinite(b)
    r_obs = _corr(a[keep], b[keep], method)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        ap = a[rng.permutation(a.shape[0])]
        k = np.isfinite(ap) & np.isfinite(b)
        null[i] = _corr(ap[k], b[k], method)

    p = (np.sum(np.abs(null) >= abs(r_obs)) + 1) / (n_perm + 1)
    return {'r': r_obs, 'p': float(p), 'n_nulls': n_perm, 'n_items': int(keep.sum()),
            'null': null, 'labels': labels}
