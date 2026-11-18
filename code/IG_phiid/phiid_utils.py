import numpy as np

import os as _os
atoms = np.load(_os.path.join(_os.path.dirname(__file__), 'phiid_keys.npy'), allow_pickle=True)

atom_pairings = [
    ['sta', 'stb'],
    ['rta', 'rtb'],
    ['xta', 'ytb'],
    ['xtb', 'yta'],
    ['xtr', 'ytr'],
    ['sta', 'stb']
]

def convert_atom_notation(atom, option=''):
    if option == 'WB':
        replacements = {
            'r': r'\{1\}\{2\}',
            'a': r'\{1\}',
            'x': r'\{1\}',
            'b': r'\{2\}',
            'y': r'\{2\}',
            's': r'\{1,2\}',
            't': r'\rightarrow',
        }
    else:
        replacements = {
            'r': r'R',
            'a': r'U_1',
            'x': r'U_1',
            'b': r'U_2',
            'y': r'U_2',
            's': r'S',
            't': r' \rightarrow ',
        }
    for old, new in replacements.items():
        atom = atom.replace(old, new)
    return f'${atom}$'

def wrangle_asymmetric_atoms(phiid, atom, atom_pairings=atom_pairings,
                             atoms=atoms,
                             make_symmetric=False):
    matrix = phiid[:, :, list(atoms).index(atom)].astype(float)
    if atom in np.array(atom_pairings).flatten():
        pair_index = None
        for pair in atom_pairings:
            if atom in pair:
                pair_index = pair[1] if atom == pair[0] else pair[0]
                break
        pair_matrix = phiid[:, :, list(atoms).index(pair_index)].astype(float)
        upper_triangle_indices = np.triu_indices_from(matrix, k=1)
        matrix[upper_triangle_indices] = pair_matrix[upper_triangle_indices]
        matrix = matrix.T 
    if make_symmetric:
        matrix = matrix + matrix.T
    return matrix # rows are sources, columns are targets