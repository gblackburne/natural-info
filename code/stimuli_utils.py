from __future__ import annotations

import csv
import functools
import json
import os
import pickle
import re
import unicodedata
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from statsmodels.stats.multitest import multipletests
    _have_statsmodels = True
except Exception:
    _have_statsmodels = False


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
repo_root = Path(__file__).resolve().parent
bids_root = Path(os.environ.get("NNDB_REPO_ROOT", repo_root / "annotations" / "bids_data")).resolve()
sem_outputs_dir = bids_root / "other_movie_annotations" / "semantic_outputs"
winner_states_pkl = repo_root / "annotations" / "nndb" / "data" / "winner_states.pkl"
norm_manifest = bids_root / "code" / "semantics" / "simple_semantic_regression" / "norm_sources.json"

results_root = repo_root / "results" / "stimuli"
normative_dir = results_root / "normative"
embedding_dir = results_root / "embeddings"

n_atoms = 10
atom_ids = list(range(1, n_atoms + 1))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_inputs() -> None:
    """Fail early and loudly rather than halfway through a permutation run."""
    missing = []
    if not winner_states_pkl.is_file():
        missing.append(str(winner_states_pkl))
    if not sem_outputs_dir.is_dir():
        missing.append(str(sem_outputs_dir))
    if not norm_manifest.is_file():
        missing.append(str(norm_manifest))
    if missing:
        raise SystemExit("Missing required inputs:\n  " + "\n  ".join(missing) +
                         "\n\nSet NNDB_REPO_ROOT if bids_data lives somewhere else.")


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def load_winner_states() -> Tuple[Dict[int, str], Dict[int, np.ndarray]]:
    with open(winner_states_pkl, "rb") as f:
        d = pickle.load(f)
    legend = {int(k): str(v) for k, v in d["legend"].items()}
    sequences = {int(k): np.asarray(v).astype(np.int64) for k, v in d["sequences"].items()}
    return legend, sequences


@functools.lru_cache(maxsize=1)
def atom_names() -> Dict[int, str]:
    return load_winner_states()[0]


@functools.lru_cache(maxsize=1)
def _length_to_movie() -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for p in sorted(sem_outputs_dir.iterdir()):
        mat = p / "semantic_matrix.npy"
        if p.is_dir() and mat.is_file():
            n = np.load(mat, mmap_mode="r").shape[0]
            if n in mapping:
                raise RuntimeError(f"two films share n_trs={n}: {mapping[n]} and {p.name}")
            mapping[n] = p.name
    return mapping


@functools.lru_cache(maxsize=1)
def participant_movie_map() -> Dict[int, str]:
    _, sequences = load_winner_states()
    lookup = _length_to_movie()
    out = {}
    for pid, seq in sequences.items():
        n_trs = len(seq) + 1
        if n_trs not in lookup:
            raise RuntimeError(f"participant {pid}: n_trs={n_trs} matches no film")
        out[pid] = lookup[n_trs]
    return out


def movie_participants() -> Dict[str, List[int]]:
    out: Dict[str, List[int]] = {}
    for pid, movie in participant_movie_map().items():
        out.setdefault(movie, []).append(pid)
    for movie in out:
        out[movie].sort()
    return out


@functools.lru_cache(maxsize=None)
def load_embeddings(movie: str, matrix: str = "semantic_matrix.npy") -> np.ndarray:
    path = sem_outputs_dir / movie / matrix
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path).astype(np.float32)


@functools.lru_cache(maxsize=None)
def load_captions(movie: str) -> list:
    path = sem_outputs_dir / movie / "captions.json"
    if not path.is_file():
        return []
    with open(path) as f:
        return json.load(f)


def caption_for_tr(movie: str, tr: int) -> dict:
    caps = load_captions(movie)
    if not (0 <= tr < len(caps)):
        return {}
    c = caps[tr]
    if c.get("global_tr_index", tr) != tr:
        for cc in caps:
            if cc.get("global_tr_index") == tr:
                c = cc
                break
    return {"dialogue": c.get("dialogue", "") or "",
            "visual_embed_text": c.get("visual_embed_text", "") or "",
            "env_audio_embed_text": c.get("env_audio_embed_text", "") or "",
            "caption": c.get("caption", "") or ""}


# --------------------------------------------------------------------------
# Alignment and the aligned stack
# --------------------------------------------------------------------------
def aligned_indices(seq_len: int, n_trs: int, offset: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    win_pos = np.arange(seq_len, dtype=np.int64)
    emb_rows = win_pos + offset
    ok = (emb_rows >= 0) & (emb_rows < n_trs)
    return win_pos[ok], emb_rows[ok]


@functools.lru_cache(maxsize=None)
def valid_tr_mask(movie: str, matrix: str = "semantic_matrix.npy") -> np.ndarray:
    """Usable TRs: not flagged failed, non-degenerate embedding, non-empty caption.

    The caption check drops intro cards and black frames, which come through as
    '0;' or an empty string and would otherwise sit at an arbitary place in the
    embedding space.
    """
    e = load_embeddings(movie, matrix)
    n = e.shape[0]
    mask = np.ones(n, dtype=bool)
    fpath = sem_outputs_dir / movie / "failed_mask.npy"
    if fpath.is_file():
        fm = np.load(fpath)
        if fm.shape[0] == n:
            mask &= ~fm.astype(bool)
    mask &= np.linalg.norm(e, axis=1) > 1e-8
    caps = load_captions(movie)
    if len(caps) == n:
        for i, c in enumerate(caps):
            cap = (c.get("caption", "") or "").strip().strip(";. ")
            if cap == "" or cap == "0" or c.get("failed", False):
                mask[i] = False
    return mask


def build_stack(offset: int = 0, matrix: str = "semantic_matrix.npy",
                exclude_failed: bool = True, movies: Optional[List[str]] = None) -> Dict[str, Dict]:
    """movie -> {emb_rows, R (movie-residualised), E (raw), onehots, pids, n_part}.

    R is the embedding with the film's mean subtracted. Every test downstream runs
    in that space, so an effect can never be film identity in disguise.
    """
    _, sequences = load_winner_states()
    per_movie = movie_participants()
    if movies is not None:
        per_movie = {m: p for m, p in per_movie.items() if m in set(movies)}
    out: Dict[str, Dict] = {}
    for movie, parts in per_movie.items():
        e_full = load_embeddings(movie, matrix)
        seq_len = len(sequences[parts[0]])
        win_pos, emb_rows = aligned_indices(seq_len, e_full.shape[0], offset)
        if exclude_failed:
            keep = valid_tr_mask(movie, matrix)[emb_rows]
            win_pos, emb_rows = win_pos[keep], emb_rows[keep]
        e = e_full[emb_rows]
        r = e - e.mean(axis=0, keepdims=True)
        t = len(emb_rows)
        onehots = []
        for pid in parts:
            atoms = sequences[pid][win_pos]
            oh = np.zeros((t, n_atoms), dtype=np.float32)
            oh[np.arange(t), atoms - 1] = 1.0
            onehots.append(oh)
        out[movie] = {"emb_rows": emb_rows, "R": r.astype(np.float32), "E": e.astype(np.float32),
                      "onehots": onehots, "pids": list(parts), "n_part": len(parts)}
    return out


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def bh_fdr(pvals) -> np.ndarray:
    if _have_statsmodels:
        return multipletests(pvals, method="fdr_bh")[1]
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = np.minimum.accumulate((p[order] * n / (np.arange(n) + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def zscore_within_movie(f: np.ndarray) -> np.ndarray:
    """Column z-score. Non-finite entries are imputed to the column mean, which
    becomes exactly 0 after centering, so undefined window edges sit neutral."""
    z = f.astype(np.float64).copy()
    for j in range(z.shape[1]):
        col = z[:, j]
        mu = np.nanmean(col) if np.isfinite(col).any() else 0.0
        col = np.where(np.isfinite(col), col, mu)
        sd = col.std()
        z[:, j] = (col - mu) / sd if sd > 1e-12 else 0.0
    return z


def compute_g(stack, fmats, rng=None, shift=False):
    """G[atom, feature] = sum of the feature over every occurence of that atom."""
    nf = next(iter(fmats.values())).shape[1]
    g = np.zeros((n_atoms, nf))
    counts = np.zeros(n_atoms)
    for movie, mv in stack.items():
        fm = fmats[movie]
        t = fm.shape[0]
        cm = np.zeros((t, n_atoms))
        for oh in mv["onehots"]:
            cm += np.roll(oh, int(rng.integers(1, t)), axis=0) if shift else oh
        g += cm.T @ fm
        counts += cm.sum(axis=0)
    return g, counts


def run_feature_test(stack, fmats, feature_names, n_perm, rng, outdir, legend,
                     prefix="", write=True):
    """Circular-shift test of every feature across atoms. Returns (z, q, means).

    The null shifts each participant's atom sequence by a random amount, which
    keeps that atom's frequency and the autocorrelation of the sequence intact
    while breaking its alignment to the film. Observed and null statistics go
    through identical pooling, so any shrinkage from pooling thousands of TRs per
    participant appears in both and cancels.
    """
    g, counts = compute_g(stack, fmats, shift=False)
    means = g / counts[:, None]
    ssb_obs = (counts[:, None] * means ** 2).sum(axis=0)

    nf = len(feature_names)
    null_means = np.empty((n_perm, n_atoms, nf))
    null_ssb = np.empty((n_perm, nf))
    for p in range(n_perm):
        gp, counts_p = compute_g(stack, fmats, rng=rng, shift=True)
        mp = gp / counts_p[:, None]
        null_means[p] = mp
        null_ssb[p] = (counts_p[:, None] * mp ** 2).sum(axis=0)
        if (p + 1) % max(1, n_perm // 10) == 0:
            print(f"  {prefix}perm {p + 1}/{n_perm}")

    mu = null_means.mean(axis=0)
    sd = null_means.std(axis=0) + 1e-12
    z = (means - mu) / sd
    p_two = (1 + (np.abs(null_means - mu) >= np.abs(means - mu)[None]).sum(axis=0)) / (n_perm + 1)
    q = bh_fdr(p_two.ravel()).reshape(p_two.shape)
    omni_p = (1 + (null_ssb >= ssb_obs[None]).sum(axis=0)) / (n_perm + 1)
    omni_q = bh_fdr(omni_p)

    if write:
        with open(outdir / "feature_atom_zscores.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["atom_id", "name", "feature", "mean_sd_units", "z", "p_perm",
                        "q_fdr", "sig_q05"])
            for a in atom_ids:
                for j, feat in enumerate(feature_names):
                    w.writerow([a, legend[a], feat, round(float(means[a - 1, j]), 5),
                                round(float(z[a - 1, j]), 3), round(float(p_two[a - 1, j]), 5),
                                round(float(q[a - 1, j]), 5), bool(q[a - 1, j] < 0.05)])
        with open(outdir / "feature_omnibus.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["feature", "SS_between", "p_perm", "q_fdr", "sig_q05"])
            for j, feat in enumerate(feature_names):
                w.writerow([feat, round(float(ssb_obs[j]), 4), round(float(omni_p[j]), 5),
                            round(float(omni_q[j]), 5), bool(omni_q[j] < 0.05)])
    return z, q, means


# --------------------------------------------------------------------------
# Embedding geometry (PERMANOVA-style omnibus, phase 1)
# --------------------------------------------------------------------------
def compute_s(stack_pca, rng=None, shift=False):
    """S[dim, atom] summed over occurrences, plus per-atom counts and SS_total."""
    d = next(iter(stack_pca.values()))["R"].shape[1]
    s = np.zeros((d, n_atoms), dtype=np.float64)
    counts = np.zeros(n_atoms, dtype=np.float64)
    ss_total = 0.0
    for mv in stack_pca.values():
        r = mv["R"]
        t = r.shape[0]
        cm = np.zeros((t, n_atoms), dtype=np.float64)
        for oh in mv["onehots"]:
            cm += np.roll(oh, int(rng.integers(1, t)), axis=0) if shift else oh
        s += r.T @ cm
        counts += cm.sum(axis=0)
        ss_total += float((cm.sum(axis=1) * np.einsum("ij,ij->i", r, r)).sum())
    return s, counts, ss_total


def ss_between(s, counts):
    nz = counts > 0
    return float((np.sum(s[:, nz] ** 2, axis=0) / counts[nz]).sum())


def pseudo_f(ssb, ss_total, n_total, k):
    ssw = ss_total - ssb
    df_b, df_w = k - 1, n_total - k
    if ssw <= 0 or df_w <= 0:
        return np.inf
    return (ssb / df_b) / (ssw / df_w)


def atom_pairs():
    return list(combinations(range(n_atoms), 2))


# --------------------------------------------------------------------------
# Lexical norms (UNITy corpora)
# --------------------------------------------------------------------------
token_re = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
word_split_re = re.compile(r"[^a-z0-9']+")


def normalize_token(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    text = text.strip().lower().replace("\u2019", "'")
    m = token_re.search(text)
    return m.group(0) if m else ""


def _read_table(path: Path, delimiter: str) -> pd.DataFrame:
    if delimiter == "tab":
        sep = "\t"
    elif delimiter in (None, "auto"):
        sep = None
    else:
        sep = delimiter
    kw = dict(sep=sep, encoding="utf-8-sig", on_bad_lines="skip")
    if sep is None:
        kw["engine"] = "python"
    else:
        kw["low_memory"] = False
    return pd.read_csv(path, **kw)


def load_norms(manifest_path: Path = None, want=None):
    """feature name -> {token: table-wide z}. Same manifest UNITy regresses on."""
    manifest_path = manifest_path or norm_manifest
    manifest = json.loads(manifest_path.read_text())
    tables_cfg = manifest["tables"]
    feats_cfg = manifest["features"]
    if want:
        feats_cfg = {k: v for k, v in feats_cfg.items() if k in set(want)}

    tables = {}
    for tname in {v["table"] for v in feats_cfg.values()}:
        cfg = tables_cfg[tname]
        df = _read_table(bids_root / cfg["path"], cfg.get("delimiter", "auto"))
        df["_tok"] = df[cfg["word_column"]].map(normalize_token)
        tables[tname] = df[df["_tok"] != ""]

    names, dicts = [], {}
    for fname, fc in feats_cfg.items():
        df = tables[fc["table"]]
        vals = pd.to_numeric(df[fc["column"]], errors="coerce")
        mu, sd = vals.mean(), vals.std()
        z = (vals - mu) / sd if sd and np.isfinite(sd) else vals * 0.0
        d = {}
        for tok, zz in zip(df["_tok"].values, z.values):
            if np.isfinite(zz) and tok not in d:
                d[tok] = float(zz)
        if d:
            names.append(fname)
            dicts[fname] = d
    return names, dicts


def tr_text(cap: dict, source: str) -> str:
    if source == "dialogue":
        return cap.get("dialogue", "") or ""
    if source == "visual":
        return cap.get("visual_embed_text", "") or ""
    if source == "env":
        return cap.get("env_audio_embed_text", "") or ""
    return " ".join([cap.get("dialogue", "") or "",
                     cap.get("visual_embed_text", "") or "",
                     cap.get("env_audio_embed_text", "") or ""])


def build_norm_features(movie, emb_rows, names, dicts, source):
    """Per TR, the mean table-wide z of whichever caption tokens matched."""
    caps = load_captions(movie)
    f = np.full((len(emb_rows), len(names)), np.nan)
    n_tokens = np.zeros(len(emb_rows))
    for r, tr in enumerate(emb_rows):
        cap = caps[tr] if tr < len(caps) else {}
        toks = [t for t in word_split_re.split(tr_text(cap, source).lower()) if t]
        n_tokens[r] = len(toks)
        if not toks:
            continue
        for j, fname in enumerate(names):
            d = dicts[fname]
            vals = [d[t] for t in toks if t in d]
            if vals:
                f[r, j] = float(np.mean(vals))
    return f, n_tokens


# --------------------------------------------------------------------------
# Semantic trajectory measures
# --------------------------------------------------------------------------
# NB :::: all these run on the independent-per-TR embedding (semantic_matrix_unfaded.npy)
# The default semantic_matrix.npy has an exponential temporal context baked in (half life ~6.2 s), 
# whose autocorr wil mess up our measures of speeed etc

def _row_norms(m):
    return np.linalg.norm(m, axis=1)


def consecutive_cosdist(m):
    out = np.full(m.shape[0], np.nan)
    if m.shape[0] < 2:
        return out
    a, b = m[:-1].astype(np.float64), m[1:].astype(np.float64)
    na, nb = _row_norms(a), _row_norms(b)
    ok = (na > 1e-8) & (nb > 1e-8)
    vals = np.full(m.shape[0] - 1, np.nan)
    vals[ok] = 1.0 - np.clip(np.einsum("ij,ij->i", a[ok], b[ok]) / (na[ok] * nb[ok]), -1, 1)
    out[1:] = vals
    return out

def _window_cosdist(m, window, future):
    """Distance from the current state to the mean of a neighbouring window."""
    out = np.full(m.shape[0], np.nan)
    norms = _row_norms(m)
    ok_rows = norms > 1e-8
    prefix = np.vstack([np.zeros((1, m.shape[1])), np.cumsum(m.astype(np.float64), axis=0)])
    prefix_n = np.concatenate([[0], np.cumsum(ok_rows.astype(np.int64))])
    for t in range(m.shape[0]):
        if future:
            lo, hi = t + 1, min(m.shape[0], t + window + 1)
        else:
            lo, hi = max(0, t - window), t
        n_valid = prefix_n[hi] - prefix_n[lo]
        if n_valid <= 0 or norms[t] <= 1e-8:
            continue
        mean_vec = (prefix[hi] - prefix[lo]) / float(n_valid)
        nm = np.linalg.norm(mean_vec)
        if nm <= 1e-8:
            continue
        out[t] = 1.0 - np.clip(float(np.dot(m[t], mean_vec) / (norms[t] * nm)), -1, 1)
    return out

def lagged_cosdist(m, lag):
    out = np.full(m.shape[0], np.nan)
    if lag <= 0 or m.shape[0] <= lag:
        return out
    a, b = m[:-lag].astype(np.float64), m[lag:].astype(np.float64)
    na, nb = _row_norms(a), _row_norms(b)
    ok = (na > 1e-8) & (nb > 1e-8)
    vals = np.full(m.shape[0] - lag, np.nan)
    if ok.any():
        vals[ok] = 1.0 - np.clip(np.einsum("ij,ij->i", a[ok], b[ok]) / (na[ok] * nb[ok]), -1, 1)
    out[lag:] = vals
    return out

def forward_lagged_cosdist(m, lag):
    back = lagged_cosdist(m, lag)
    out = np.full_like(back, np.nan)
    if lag < len(back):
        out[:-lag] = back[lag:]
    return out

def _rolling(x, before=0, after=0, stat="mean"):
    out = np.full(len(x), np.nan)
    finite = np.isfinite(x)
    vals = np.where(finite, x, 0.0)
    p1 = np.concatenate([[0.0], np.cumsum(vals)])
    p2 = np.concatenate([[0.0], np.cumsum(vals ** 2)])
    pn = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])
    for t in range(len(x)):
        lo, hi = max(0, t - before), min(len(x), t + after + 1)
        n = pn[hi] - pn[lo]
        if stat == "mean" and n > 0:
            out[t] = (p1[hi] - p1[lo]) / float(n)
        elif stat == "sd" and n > 1:
            mean = (p1[hi] - p1[lo]) / float(n)
            mean2 = (p2[hi] - p2[lo]) / float(n)
            out[t] = np.sqrt(max(0.0, mean2 - mean ** 2))
    return out

def future_mean(x, n_future):
    out = np.full(len(x), np.nan)
    finite = np.isfinite(x)
    vals = np.where(finite, x, 0.0)
    p1 = np.concatenate([[0.0], np.cumsum(vals)])
    pn = np.concatenate([[0], np.cumsum(finite.astype(np.int64))])
    for t in range(len(x)):
        lo, hi = t + 1, min(len(x), t + n_future + 1)
        n = pn[hi] - pn[lo]
        if n > 0:
            out[t] = (p1[hi] - p1[lo]) / float(n)
    return out

def window_coherence(m, start_offset, end_offset):
    """Norm of the mean unit vector over a window. Near 1 means the film is parked."""
    n = m.shape[0]
    out = np.full(n, np.nan)
    norms = _row_norms(m)
    ok = norms > 1e-8
    u = np.zeros_like(m, dtype=np.float64)
    u[ok] = m[ok].astype(np.float64) / norms[ok, None]
    prefix = np.vstack([np.zeros((1, m.shape[1])), np.cumsum(u, axis=0)])
    prefix_n = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])
    for t in range(n):
        lo, hi = max(0, t + start_offset), min(n - 1, t + end_offset)
        if hi < lo:
            continue
        n_valid = prefix_n[hi + 1] - prefix_n[lo]
        if n_valid <= 0:
            continue
        out[t] = float(np.linalg.norm((prefix[hi + 1] - prefix[lo]) / float(n_valid)))
    return out

def recurrence_distance(m, start_lag, end_lag, future=False):
    """Distance to the nearest matching state in a lag band.

    start_lag of 4 keeps this from just rediscovering local smoothness. What we
    want is a genuine callback, not the fact that t-1 looks like t.
    """
    n = m.shape[0]
    dist = np.full(n, np.nan)
    norms = _row_norms(m)
    for t in range(n):
        if norms[t] <= 1e-8:
            continue
        if future:
            idx = np.arange(t + start_lag, min(n, t + end_lag + 1))
        else:
            idx = np.arange(max(0, t - end_lag), t - start_lag + 1)
        if idx.size == 0:
            continue
        ok = norms[idx] > 1e-8
        if not ok.any():
            continue
        idx = idx[ok]
        cos = np.clip((m[idx].astype(np.float64) @ m[t].astype(np.float64)) /
                      (norms[idx] * norms[t]), -1, 1)
        dist[t] = float((1.0 - cos).min())
    return dist


measure_descriptions = {
    "past6_deviation": "Distance from the mean of the previous 6 TRs.",
    "past12_deviation": "Distance from the mean of the previous 12 TRs.",
    "past24_deviation": "Distance from the mean of the previous 24 TRs.",
    "future6_deviation": "Distance from the mean of the next 6 TRs.",
    "future12_deviation": "Distance from the mean of the next 12 TRs.",
    "lag3_distance": "Distance to the state 3 TRs ago.",
    "lag6_distance": "Distance to the state 6 TRs ago.",
    "lag12_distance": "Distance to the state 12 TRs ago.",
    "past6_speed": "Mean step size over the previous 6 TRs.",
    "past12_speed": "Mean step size over the previous 12 TRs.",
    "past24_speed": "Mean step size over the previous 24 TRs.",
    "next3_speed": "Mean step size over the next 3 TRs.",
    "next6_speed": "Mean step size over the next 6 TRs.",
    "next12_speed": "Mean step size over the next 12 TRs.",
    "speed_sd_past6": "SD of step size over the previous 6 TRs.",
    "speed_sd_past12": "SD of step size over the previous 12 TRs.",
    "speed_sd_next6": "SD of step size over the next 6 TRs.",
    "coherence_past6": "Coherence of the previous 6 states.",
    "coherence_past12": "Coherence of the previous 12 states.",
    "coherence_future6": "Coherence of the next 6 states.",
    "coherence_future12": "Coherence of the next 12 states.",
    "coherence_local13": "Coherence of the 13 states centred on this TR.",
    "recurrence_past30_dist": "Distance to the nearest match 4-30 TRs back.",
    "recurrence_past120_dist": "Distance to the nearest match 4-120 TRs back.",
    "recurrence_future30_dist": "Distance to the nearest match 4-30 TRs ahead.",
    "recurrence_future120_dist": "Distance to the nearest match 4-120 TRs ahead.",
    "coherence_asym_3": "Future-3 coherence minus past-3 coherence.",
    "coherence_asym_6": "Future-6 coherence minus past-6 coherence.",
    "coherence_asym_12": "Future-12 coherence minus past-12 coherence.",
    "coherence_asym_24": "Future-24 coherence minus past-24 coherence.",
    "coherence_change_7": "Change in centred 7-TR coherence from the prior TR.",
    "coherence_change_13": "Change in centred 13-TR coherence from the prior TR.",
    "context_asym_3": "Future-3 deviation minus past-3 deviation.",
    "context_asym_6": "Future-6 deviation minus past-6 deviation.",
    "context_asym_12": "Future-12 deviation minus past-12 deviation.",
    "context_asym_24": "Future-24 deviation minus past-24 deviation.",
    "lag_asym_3": "Distance to the state 3 TRs ahead minus 3 TRs back.",
    "lag_asym_6": "Distance to the state 6 TRs ahead minus 6 TRs back.",
    "speed_vs_past3": "Current step size minus the previous 3-TR mean.",
    "speed_vs_past6": "Current step size minus the previous 6-TR mean.",
    "speed_vs_past12": "Current step size minus the previous 12-TR mean.",
    "speed_vs_past24": "Current step size minus the previous 24-TR mean.",
    "speed_vs_next3": "Current step size minus the next 3-TR mean.",
    "speed_vs_next6": "Current step size minus the next 6-TR mean.",
    "speed_vs_next12": "Current step size minus the next 12-TR mean.",
}

measure_names = list(measure_descriptions)


def extract_measures(movie, emb_rows, matrix="semantic_matrix.npy"):
    """[len(emb_rows), 45] trajectory measures for one film."""
    faded = load_embeddings(movie, matrix)
    n_trs, dim = faded.shape
    path = sem_outputs_dir / movie / "semantic_matrix_unfaded.npy"
    e = np.load(path) if path.is_file() else np.zeros((n_trs, dim), dtype=np.float32)
    if e.shape[0] != n_trs:
        e = np.zeros((n_trs, dim), dtype=np.float32)

    speed = consecutive_cosdist(e)
    past = {w: _window_cosdist(e, w, future=False) for w in (3, 6, 12, 24)}
    fut = {w: _window_cosdist(e, w, future=True) for w in (3, 6, 12, 24)}
    coh_past = {w: window_coherence(e, -w, -1) for w in (3, 6, 12, 24)}
    coh_fut = {w: window_coherence(e, 1, w) for w in (3, 6, 12, 24)}

    local = {}
    for span, half in ((7, 3), (13, 6)):
        c = window_coherence(e, -half, half)
        ch = np.full(n_trs, np.nan)
        ch[1:] = np.diff(c)
        local[span] = (c, ch)

    vals = {
        "past6_deviation": past[6], "past12_deviation": past[12], "past24_deviation": past[24],
        "future6_deviation": fut[6], "future12_deviation": fut[12],
        "lag3_distance": lagged_cosdist(e, 3), "lag6_distance": lagged_cosdist(e, 6),
        "lag12_distance": lagged_cosdist(e, 12),
        "past6_speed": _rolling(speed, before=6), "past12_speed": _rolling(speed, before=12),
        "past24_speed": _rolling(speed, before=24),
        "next3_speed": future_mean(speed, 3), "next6_speed": future_mean(speed, 6),
        "next12_speed": future_mean(speed, 12),
        "speed_sd_past6": _rolling(speed, before=6, stat="sd"),
        "speed_sd_past12": _rolling(speed, before=12, stat="sd"),
        "speed_sd_next6": _rolling(speed, after=6, stat="sd"),
        "coherence_past6": coh_past[6], "coherence_past12": coh_past[12],
        "coherence_future6": coh_fut[6], "coherence_future12": coh_fut[12],
        "coherence_local13": local[13][0],
        "recurrence_past30_dist": recurrence_distance(e, 4, 30),
        "recurrence_past120_dist": recurrence_distance(e, 4, 120),
        "recurrence_future30_dist": recurrence_distance(e, 4, 30, future=True),
        "recurrence_future120_dist": recurrence_distance(e, 4, 120, future=True),
        "coherence_asym_3": coh_fut[3] - coh_past[3],
        "coherence_asym_6": coh_fut[6] - coh_past[6],
        "coherence_asym_12": coh_fut[12] - coh_past[12],
        "coherence_asym_24": coh_fut[24] - coh_past[24],
        "coherence_change_7": local[7][1], "coherence_change_13": local[13][1],
        "context_asym_3": fut[3] - past[3], "context_asym_6": fut[6] - past[6],
        "context_asym_12": fut[12] - past[12], "context_asym_24": fut[24] - past[24],
        "lag_asym_3": forward_lagged_cosdist(e, 3) - lagged_cosdist(e, 3),
        "lag_asym_6": forward_lagged_cosdist(e, 6) - lagged_cosdist(e, 6),
        "speed_vs_past3": speed - _rolling(speed, before=3),
        "speed_vs_past6": speed - _rolling(speed, before=6),
        "speed_vs_past12": speed - _rolling(speed, before=12),
        "speed_vs_past24": speed - _rolling(speed, before=24),
        "speed_vs_next3": speed - future_mean(speed, 3),
        "speed_vs_next6": speed - future_mean(speed, 6),
        "speed_vs_next12": speed - future_mean(speed, 12),
    }
    return np.column_stack([vals[n][emb_rows] for n in measure_names]).astype(np.float64)


# --------------------------------------------------------------------------
# Families
# --------------------------------------------------------------------------
# Sign is +1 when the measure already goes up with construct, -1 when it need flippin

families = {
    "semantic_flux": (
        ("past6_deviation", 1.0), ("past12_deviation", 1.0), ("past24_deviation", 1.0),
        ("future6_deviation", 1.0), ("future12_deviation", 1.0),
        ("lag3_distance", 1.0), ("lag6_distance", 1.0), ("lag12_distance", 1.0),
        ("past6_speed", 1.0), ("past12_speed", 1.0), ("past24_speed", 1.0),
        ("next3_speed", 1.0), ("next6_speed", 1.0), ("next12_speed", 1.0),
        ("speed_sd_past6", 1.0), ("speed_sd_past12", 1.0), ("speed_sd_next6", 1.0),
        ("coherence_past6", -1.0), ("coherence_past12", -1.0),
        ("coherence_future6", -1.0), ("coherence_future12", -1.0),
        ("coherence_local13", -1.0),
        ("recurrence_past30_dist", 1.0), ("recurrence_past120_dist", 1.0),
        ("recurrence_future30_dist", 1.0), ("recurrence_future120_dist", 1.0),
    ),
    "coherence_rise": (
        ("coherence_asym_3", 1.0), ("coherence_asym_6", 1.0),
        ("coherence_asym_12", 1.0), ("coherence_asym_24", 1.0),
        ("coherence_change_7", 1.0), ("coherence_change_13", 1.0),
    ),
    "context_asymmetry": (
        ("context_asym_3", -1.0), ("context_asym_6", -1.0),
        ("context_asym_12", -1.0), ("context_asym_24", -1.0),
        ("lag_asym_3", -1.0), ("lag_asym_6", -1.0),
    ),
    "speed_surprise": (
        ("speed_vs_past3", 1.0), ("speed_vs_past6", 1.0),
        ("speed_vs_past12", 1.0), ("speed_vs_past24", 1.0),
        ("speed_vs_next3", 1.0), ("speed_vs_next6", 1.0), ("speed_vs_next12", 1.0),
    ),
}

def orient(zraw, members):
    """Pick out a family's measures and flip signs so + is always more of it."""
    cols = [measure_names.index(n) for n, _ in members]
    signs = np.array([s for _, s in members])
    return {m: zraw[m][:, cols] * signs for m in zraw}

def family_typicality(zraw, members):
    """Each member's mean correlation to the rest of its family."""
    block = orient(zraw, members)
    pooled = np.vstack([block[m] for m in sorted(block)])
    corr = np.corrcoef(pooled, rowvar=False)
    np.fill_diagonal(corr, np.nan)
    return np.nanmean(corr, axis=1)

def build_loo(block):
    """Column k becomes the mean of every column except k."""
    n = block.shape[1]
    return (block.sum(axis=1, keepdims=True) - block) / float(n - 1)

def axis_matrices(zraw, mode="mean"):
    """One column per family: the member mean, or the single most typical member."""
    picks = {}
    for fam, members in families.items():
        typ = family_typicality(zraw, members)
        if mode == "mean":
            picks[fam] = (f"mean of {len(members)} measures", tuple(members),
                          float(np.nanmean(typ)), typ)
        else:
            k = int(np.nanargmax(typ))
            picks[fam] = (members[k][0], (members[k],), float(typ[k]), typ)
    mats = {}
    for m in zraw:
        cols = []
        for fam in families:
            used = picks[fam][1]
            idx = [measure_names.index(n) for n, _ in used]
            sg = np.array([s for _, s in used])
            cols.append((zraw[m][:, idx] * sg).mean(axis=1))
        mats[m] = np.column_stack(cols)
    return picks, mats
