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
                         '..', 'fMRI_general', 'atlases', 'schaefer_extend_atlases')
n_regions = 1054
subjects = np.arange(1, 87)
regions = np.arange(n_regions)
sfreq = 1
tau = 1

phiid_atoms = list(np.load('IG_phiid/phiid_keys.npy', allow_pickle=True))

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

film_p_dict = {'500_days_of_summer': np.arange(1, 21),
                'citizenfour': np.arange(21, 39),
                'the_usual_suspects': np.arange(39, 45),
                'pulp_fiction': np.arange(45, 51),
                'the_shawshank_redemption': np.arange(51, 57),
                'the_prestige': np.arange(57, 63),
                'back_to_the_future': np.arange(63, 69),
                'split': np.arange(69, 75),
                'little_miss_sunshine': np.arange(75, 81),
                '12_years_a_slave': np.arange(81, 87)}
film_p_dict_r = {perp: film for film, perps in film_p_dict.items() for perp in perps}


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
