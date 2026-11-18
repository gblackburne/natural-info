# Rather than resampling annotation to 1.5 s, resampled brain up toi the 1 Hz grid. 

import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

here = Path(__file__).resolve().parent
root = here.parent
sys.path.insert(0, str(root)) # the measure bank and the norm tables live atroot
sys.path.insert(0, str(here)) # replication misc.py must shadow the one at the root

import misc
import stimuli_utils as _su

n_atoms = _su.n_atoms
atom_ids = _su.atom_ids
sem_outputs_dir = _su.sem_outputs_dir
norm_manifest = _su.norm_manifest

ensure_dir = _su.ensure_dir
bh_fdr = _su.bh_fdr
zscore_within_movie = _su.zscore_within_movie
compute_g = _su.compute_g
run_feature_test = _su.run_feature_test
compute_s = _su.compute_s
ss_between = _su.ss_between
pseudo_f = _su.pseudo_f
atom_pairs = _su.atom_pairs
load_norms = _su.load_norms
tr_text = _su.tr_text
load_embeddings = _su.load_embeddings
load_captions = _su.load_captions
valid_tr_mask = _su.valid_tr_mask
caption_for_tr = _su.caption_for_tr
extract_measures = _su.extract_measures
measure_names = _su.measure_names
measure_descriptions = _su.measure_descriptions
families = _su.families
orient = _su.orient
family_typicality = _su.family_typicality
build_loo = _su.build_loo
axis_matrices = _su.axis_matrices
word_split_re = _su.word_split_re


parcellation = misc.parcellation
movie = 'backtothefuture'

winner_states_pkl = (here / 'nndb2_results' / 'dynamics_winner' / parcellation /
                     'export' / 'winner_states.pkl')

results_root = here / 'nndb2_results' / 'stimuli' # anchored to replication/, not me launch dir
normative_dir = results_root / 'normative'
embedding_dir = results_root / 'embeddings'

_meta = json.loads((sem_outputs_dir / movie / 'semantic_metadata.json').read_text())
annotation_tr = float(_meta['core_duration'])
scan_tr = float(misc.TR)
tr_ratio = scan_tr / annotation_tr


def check_inputs() -> None:
    """Fail early and loudly rather than halfway through a permutation run."""
    missing = []
    if not winner_states_pkl.is_file():
        missing.append(str(winner_states_pkl))
    if not (sem_outputs_dir / movie).is_dir():
        missing.append(str(sem_outputs_dir / movie))
    if not norm_manifest.is_file():
        missing.append(str(norm_manifest))
    if missing:
        raise SystemExit('Missing required inputs:\n  ' + '\n  '.join(missing) +
                         '\n\nSet NNDB_REPO_ROOT if bids_data lives somewhere else.')



def load_winner_states() -> Tuple[Dict[int, str], Dict[int, np.ndarray]]:
    with open(winner_states_pkl, 'rb') as f:
        d = pickle.load(f)
    legend = {int(k): str(v) for k, v in d['legend'].items()}
    # nndb2 subject ids are zero padded strings ('04'), the original run used ints
    sequences = {int(k): np.asarray(v).astype(np.int64) for k, v in d['sequences'].items()}
    return legend, sequences


def atom_names() -> Dict[int, str]:
    return load_winner_states()[0]


def participant_movie_map() -> Dict[int, str]:
    # every NNDb-2 participant watched the same film, so there is nothing to infer
    return {pid: movie for pid in sorted(load_winner_states()[1])}


def movie_participants() -> Dict[str, List[int]]:
    return {movie: sorted(load_winner_states()[1])}


def aligned_indices(seq_len: int, n_trs: int, offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Atom window -> annotation row, at 1.5 s per window on a 1 s grid.

    Window i is the tau=1 transition from TR i to TR i+1, so it opens at scan time
    tr_ratio * i. Taking the row it opens on keeps the original run's convention,
    where window i was read against row i. offset is in annotation rows, which is to
    say seconds, and stays in seconds at either TR: the -5 of the published run is a
    5 s haemodynamic lag here as well, not 5 scan TRs.
    """
    win_pos = np.arange(seq_len, dtype=np.int64)
    emb_rows = np.floor(win_pos * tr_ratio).astype(np.int64) + offset
    ok = (emb_rows >= 0) & (emb_rows < n_trs)
    return win_pos[ok], emb_rows[ok]


def caption_spans(win_pos: np.ndarray, n_trs: int, offset: int = 0,
                  mode: str = 'span') -> List[np.ndarray]:
    """Annotation rows each atom window overlaps, for the text side.

    A 1.5 s window always straddles two of the 1 s caption bins, and taking only the
    row it opens on would throw away roughly a third of the film's dialogue. 'span'
    pools every row the window touches, 'single' keeps the original run's one row per
    window and is the sensitivity check.
    """
    if mode not in ('span', 'single'):
        raise ValueError(f"caption mode must be 'span' or 'single', got {mode!r}")
    keep = valid_tr_mask(movie)
    out = []
    for i in win_pos:
        if mode == 'single':
            rows = np.array([int(np.floor(i * tr_ratio)) + offset], dtype=np.int64)
        else:
            lo = int(np.floor(i * tr_ratio)) + offset
            hi = int(np.ceil((i + 1) * tr_ratio)) + offset
            rows = np.arange(lo, hi, dtype=np.int64)
        rows = rows[(rows >= 0) & (rows < n_trs)]
        out.append(rows[keep[rows]] if rows.size else rows)
    return out


def build_stack(offset: int = 0, matrix: str = 'semantic_matrix.npy',
                exclude_failed: bool = True, caption_mode: str = 'span',
                movies: Optional[List[str]] = None) -> Dict[str, Dict]:
    """movie -> {emb_rows, cap_rows, R, E, onehots, pids, n_part}.

    Same shape as the original stack so the analysis scripts read identically. R is
    still the film-mean-residualised embedding, which with a single film is plain
    centering: the control it provided against film identity has no work to do here,
    and the claim narrows to within Back to the Future accordingly.
    """
    _, sequences = load_winner_states()
    parts = movie_participants()[movie]
    if movies is not None and movie not in set(movies):
        return {}

    e_full = load_embeddings(movie, matrix)
    seq_len = len(sequences[parts[0]])
    win_pos, emb_rows = aligned_indices(seq_len, e_full.shape[0], offset)
    if exclude_failed:
        keep = valid_tr_mask(movie, matrix)[emb_rows]
        win_pos, emb_rows = win_pos[keep], emb_rows[keep]
    cap_rows = caption_spans(win_pos, e_full.shape[0], offset, caption_mode)

    e = e_full[emb_rows]
    r = e - e.mean(axis=0, keepdims=True)
    t = len(emb_rows)
    onehots = []
    for pid in parts:
        atoms = sequences[pid][win_pos]
        oh = np.zeros((t, n_atoms), dtype=np.float32)
        oh[np.arange(t), atoms - 1] = 1.0
        onehots.append(oh)
    return {movie: {'emb_rows': emb_rows, 'cap_rows': cap_rows,
                    'R': r.astype(np.float32), 'E': e.astype(np.float32),
                    'onehots': onehots, 'pids': list(parts), 'n_part': len(parts)}}


def build_norm_features(movie_name, cap_rows, names, dicts, source):
    """Per atom window, the mean table-wide z of whichever caption tokens matched.

    Takes the row spans from build_stack rather than single rows, so a window that
    straddles two caption bins is scored on the words from both.
    """
    caps = load_captions(movie_name)
    f = np.full((len(cap_rows), len(names)), np.nan)
    n_tokens = np.zeros(len(cap_rows))
    for r, rows in enumerate(cap_rows):
        text = ' '.join(tr_text(caps[tr] if tr < len(caps) else {}, source) for tr in rows)
        toks = [t for t in word_split_re.split(text.lower()) if t]
        n_tokens[r] = len(toks)
        if not toks:
            continue
        for j, fname in enumerate(names):
            d = dicts[fname]
            vals = [d[t] for t in toks if t in d]
            if vals:
                f[r, j] = float(np.mean(vals))
    return f, n_tokens
