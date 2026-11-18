# %%
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os, sys
import cmasher as cmr

sys.path.insert(0, '.')
sys.path.insert(0, os.path.join('.', 'IG_phiid'))
import misc
from IG_phiid.gaussian_phiid import ig_phiid_gaussian
from IG_phiid.phiid_utils import convert_atom_notation

font_path = '/home/george/Research/fonts/Instrument_Sans/static/InstrumentSans-Regular.ttf'
if 'Instrument Sans' not in [f.name for f in fm.fontManager.ttflist]:
    fm.fontManager.addfont(font_path)
plt.rcParams.update({
    'font.family': 'Instrument Sans',
    'figure.dpi': 300,
    'savefig.dpi': 800,
    'font.size': 15,
    'text.usetex': False,
    'mathtext.fontset': 'cm',
})


# %%
from IG_phiid.phiid_utils import convert_atom_notation

def make_cov_joint(A):
    """Joint [past, future] covariance for unit-variance VAR(1) with transition A.
    Variable order: [X_past, Y_past, X_future, Y_future] — matches get_cov().
    Requires max singular value of A < 1.
    """
    return np.block([[np.eye(2), A.T], [A, np.eye(2)]])


def phiid_from_A(A):
    """All 16 PhiID atoms (bits) for a 2-var VAR(1), ordered by misc.phiid_atoms."""
    sv_max = np.linalg.svd(A, compute_uv=False)[0]
    if sv_max >= 1.0:
        raise ValueError(f'Max singular value {sv_max:.3f} >= 1')
    cov = make_cov_joint(A)
    result = ig_phiid_gaussian(cov, as_dict=True)
    return np.array([result[k] for k in misc.phiid_atoms])


# Weights each atom by its squared value: w_k = a_k^2.  Squaring removes the
# sign, so positive and negative atoms contribute equally by magnitude, but
# unlike Shannon entropy the measure penalises dominance quadratically rather
# than logarithmically.  One atom ten times larger than the rest contributes
# 100x to the concentration term, making PR far more sensitive to whether a
# single atom type is carrying most of the information.

def comp_diversity(atoms):
    # i.e atomic participation ratio or Renyi-2 diversity.
    # PR = (1 / sum_k p_k^2) / n,  where p_k = a_k^2 / sum_j a_j^2
    # Range: 1/n (all mass on one atom) to 1 (uniform across all atoms).
    w = atoms ** 2
    s = w.sum()
    if s == 0:
        return 1.0
    p = w / s
    return (1.0 / np.sum(p ** 2)) / len(atoms)


# atom display ordering: group structure, coloured by group
atom_idx_order = []
atom_col_list = []
atom_lbl_list = []
for grp in misc.atom_order:
    for idx in misc.atoms[grp]:
        atom_idx_order.append(idx)
        atom_col_list.append(misc.atoms_colors[grp])
        atom_lbl_list.append(convert_atom_notation(misc.phiid_atoms[idx]))


# %%
rng = np.random.default_rng(7)
A_rand_raw = rng.standard_normal((2, 2))
sv_r = np.linalg.svd(A_rand_raw, compute_uv=False)[0]
A_rand = A_rand_raw * (0.85 / sv_r)

# a = self-coupling (diagonal of A), b = cross-coupling (off-diagonal). The noise
# correlation is zero here (the past covariance block is the identity), so c is
# left free for that convention.
canonical = {
    'Self': 0.7 * np.eye(2),
    'Cross': np.array([[0.3, 0.6], [0.6, 0.3]]),
    'Cross Inhibitory': np.array([[0.3, -0.6], [-0.6, 0.3]]),
    'Cross Asymmetric': np.array([[0.4, 0.0], [0.6, 0.4]]),
    'Cross Exclusive': np.array([[0.0, 0.8], [0.8, 0.0]]),
    'Random': A_rand,
}

# coefficients are quoted in the figure caption rather than the panel titles
coefficients = {
    'Self': 'a=0.7, b=0.0',
    'Cross': 'a=0.3, b=0.6',
    'Cross Inhibitory': 'a=0.3, b=-0.6',
    'Cross Asymmetric': 'a=0.4, b=0.6 on X->Y only',
    'Cross Exclusive': 'a=0.0, b=0.8',
    'Random': 'rescaled to max singular value 0.85',
}

for name, A in canonical.items():
    sv = np.linalg.svd(A, compute_uv=False)[0]
    print(f'{name:<18}  max_sv = {sv:.3f}   ({coefficients[name]})')

system_atoms = {name: phiid_from_A(A) for name, A in canonical.items()}
system_diversity = {name: comp_diversity(atoms) for name, atoms in system_atoms.items()}
system_mi = {name: float(np.sum(atoms)) for name, atoms in system_atoms.items()}

for name in canonical:
    print(f'{name:<18}  D = {system_diversity[name]:.3f}   MI = {system_mi[name]:.4f} bits')


# %%
from IG_phiid.tpm_utils import phiid_matrix

atom_labels = [r'$R$', r'$U_X$', r'$U_Y$', r'$S$']

fig, axes = plt.subplots(2, 3, figsize=(11, 6))
axes = axes.flatten()

for ax, (name, atoms) in zip(axes, system_atoms.items()):
    matrix = phiid_matrix(dict(zip(misc.phiid_atoms, atoms)))
    lim = max(float(np.nanmax(np.abs(matrix))), 1e-9)
    im = ax.imshow(matrix, cmap=cmr.horizon_r,
                   vmin=-lim, vmax=lim, aspect='equal')

    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(atom_labels)
    ax.set_yticklabels(atom_labels)
    if ax.get_subplotspec().is_last_row():
        ax.set_xlabel(r'Future ($\beta$)', labelpad=2)
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel(r'Past ($\alpha$)', labelpad=2)
    ax.tick_params(length=0)
    for j in range(3):
        ax.axhline(j + 0.5, color='black', linestyle='-', linewidth=0.5)
        ax.axvline(j + 0.5, color='black', linestyle='-', linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$I_{\partial}^{\alpha\rightarrow\beta}$', fontsize=plt.rcParams['font.size'], labelpad=2)
    cbar.ax.tick_params(labelsize=plt.rcParams['font.size'],
                        length=2)

    D = system_diversity[name]
    ax.set_title(rf'{name}' '\n' rf'$D$ = {D:.2f}', pad=5, fontsize=plt.rcParams['font.size'],
                 color="#3F3F3F")

plt.tight_layout()
plt.savefig('plots/toy_diversity.png', bbox_inches='tight', dpi=500)


# %%
# autocorrelation strength sweep
alphas = np.linspace(0.01, 0.95, 40)
div_ar, mi_ar = [], []
for a in alphas:
    at = phiid_from_A(a * np.eye(2))
    div_ar.append(comp_diversity(at))
    mi_ar.append(float(np.sum(at)))

# symmetric coupling sweep (fixed a=0.3)
couplings = np.linspace(0.01, 0.64, 40)
div_sym, mi_sym = [], []
for c in couplings:
    A = np.array([[0.3, c], [c, 0.3]])
    at = phiid_from_A(A)
    div_sym.append(comp_diversity(at))
    mi_sym.append(float(np.sum(at)))

# asymmetric X->Y sweep (fixed a=0.4)
c_asym = np.linspace(0.0, 0.78, 40)
div_asym, mi_asym = [], []
for c in c_asym:
    A = np.array([[0.4, 0.0], [c, 0.4]])
    try:
        at = phiid_from_A(A)
        div_asym.append(comp_diversity(at))
        mi_asym.append(float(np.sum(at)))
    except ValueError:
        div_asym.append(np.nan)
        mi_asym.append(np.nan)


# %%
n_rand = 100
strengths = np.linspace(0.05, 0.95, 10)
rng2 = np.random.default_rng(42)

rand_div = {s: [] for s in strengths}
rand_mi = {s: [] for s in strengths}

for s in strengths:
    count = 0
    while count < n_rand:
        A_raw = rng2.standard_normal((2, 2))
        sv_max = np.linalg.svd(A_raw, compute_uv=False)[0]
        A = A_raw * (s / sv_max)
        try:
            at = phiid_from_A(A)
            rand_div[s].append(comp_diversity(at))
            rand_mi[s].append(float(np.sum(at)))
            count += 1
        except Exception:
            pass

all_mi_r = np.concatenate([rand_mi[s] for s in strengths])
all_div_r = np.concatenate([rand_div[s] for s in strengths])
all_sv_r = np.concatenate([np.full(len(rand_div[s]), s) for s in strengths])


# %%
col_ar = "#C10B0B"
col_sym = "#1E3F6B"
col_asym = '#32A1C2'

fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)

ax = axes[0]
#ax.plot(alphas, div_ar, color=col_ar, lw=2, label='Self', marker='o', ms=3)
ax.plot(couplings, div_sym, color=col_sym, lw=2, label='Cross', marker='s', ms=3)
ax.plot(c_asym, div_asym, color=col_asym, lw=2, label='Cross Asymmetric', marker='^', ms=3)
ax.set_xlabel('Coupling')
ax.set_ylabel(r'$D$')
ax.set_ylim(0, 1.05)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False, loc='lower left')
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
ax.yaxis.set_major_locator(plt.MaxNLocator(3))

ax = axes[1]
sc = ax.scatter(all_mi_r, all_div_r, c=all_sv_r, cmap='gist_stern_r', s=15)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(r'$\sigma_{\max}(A)$')
ax.set_xlabel(r'$\mathrm{TDMI}$ (bits)')
ax.set_ylim(0, 1.05)
ax.spines[['top', 'right']].set_visible(False)
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
ax.yaxis.set_major_locator(plt.MaxNLocator(3))

# correlate diversity with TDMI for the random systems
corr = np.corrcoef(all_mi_r, all_div_r)[0, 1]
print(f'{corr:.3f}')

plt.tight_layout()
plt.savefig('plots/toy_diversity_sweep.png', bbox_inches='tight', dpi=500)
