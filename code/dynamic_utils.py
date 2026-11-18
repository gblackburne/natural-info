import numpy as np


import sys
sys.path.insert(0, '.')
from pyNuMIT.SS.CSER import StateSpaceEntropyRate


def normalise(data, axis=1):
    """Z-score data along the given axis."""
    mean = np.nanmean(data, axis=axis, keepdims=True)
    std = np.nanstd(data, axis=axis, keepdims=True)
    std[std == 0] = 1
    return (data - mean) / std


def to_features(data):
    """Reshape (n_regions, n_time, n_atoms) -> (n_time, n_regions*n_atoms)."""
    n_regions, _, n_atoms = data.shape
    return data.transpose(1, 0, 2).reshape(-1, n_regions * n_atoms)


def split_by_lengths(X, lengths):
    """Split a concatenated (sum_T, d) array back into a list of per-sequence arrays."""
    ends = np.cumsum(lengths)
    starts = np.concatenate([[0], ends[:-1]])
    return [X[s:e] for s, e in zip(starts, ends)]


def entropy_rate(data, Fs=1, downsampling="no", varmomax=20):
    """
    Complexity via State-Space Entropy Rate

    Parameters
    ----------
    data: 1D array

    Fs: float
        Sampling frequency of the data (Hz)
    downsampling: str
        Downsampling method
    varmomax: int
        Maximum VAR model order (default is 20).

    Returns
    ------- 
    er: float
        Entropy rate of the data

    """
    try:
        data = np.asarray(data)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        er = StateSpaceEntropyRate(data, Fs, downsampling=downsampling, varmomax=varmomax)[0]
    except Exception as e:
        print(f"Error calculating entropy rate: {e}")
        er = np.nan
    return er


def state_dynamics(state_seq, n_states, TR=1):
    """
    Calculate dynamics of each state

    Parameters
    ----------
    state_seq: array
        State sequence
    n_states: int
        Number of states
    TR: float
        Repetition time (seconds)

    Returns
    -------
    f_occ: dict
        Fractional occupancy of each state
    dt: dict
        Mean dwell time for each state
    vt: dict
        Variance of dwell time for each state
    ar: dict
        Appearance rate for each state
    """
    state_seq = np.array(state_seq)
    
    # fractional occupancy of each state
    f_occ = {}
    for state in range(1, n_states+1):
        f_occ[state] = np.sum(state_seq == state) / len(state_seq)
    
    dt = {} # mean dwell time for each state
    vt = {} # variance of dwell time for each state
    ar = {} # appearance rate for each state
    change_indices = np.where(np.diff(state_seq) != 0)[0] + 1
    change_indices = np.concatenate(([0], change_indices, [len(state_seq)]))
    dwell_times = np.diff(change_indices)
    for state in range(1, n_states+1):
        mask = state_seq[change_indices[:-1]] == state
        dt[state] = np.mean(dwell_times[mask] if np.any(mask) else 0)
        vt[state] = np.var(dwell_times[mask] if np.any(mask) else 0)
        ar[state] = np.sum(mask) / len(state_seq)

    dt = {k: v * TR for k, v in dt.items()} # Convert to seconds
    vt = {k: v * TR for k, v in vt.items()} # Convert to seconds

    # double check that all states are accounted for
    for state in range(1, n_states+1):
        if state not in f_occ:
            f_occ[state] = 0
        if state not in dt:
            dt[state] = 0
        if state not in vt:
            vt[state] = 0
        if state not in ar:
            ar[state] = 0

    return f_occ, dt, vt, ar

def generate_tpm(state_seq, n_states):
    """
    Generate transition probability matrix

    Parameters
    ----------
    state_seq: array
        State sequence
    n_states: int
        Number of states

    Returns
    -------
    tpm: array
        Transition probability matrix (n_states, n_states)
        (current state, next state)

    """
    
    state_seq = np.round(state_seq) # ensure
    
    tpm = np.zeros((n_states, n_states))

    for i in range(n_states):
        for j in range(n_states):
            tpm[i, j] = np.nansum((state_seq[:-1] == i+1) & (state_seq[1:] == j+1))
    tpm = np.nan_to_num(tpm)
    np.fill_diagonal(tpm, 0)
    row_sums = tpm.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore'):
        tpm = np.where(row_sums > 0, tpm / row_sums, np.nan)

    return tpm

def row_zscore(X):
    """
    Z-score each sample (row) across its features.

    For correlation-distance k-means: once each row has zero mean and unit
    variance the squared Euclidean distance between two rows equals
    2 * n_features * (1 - r), where r is their Pearson correlation. Euclidean
    k-means on row-z-scored data is therefore clustering by 1 - correlation.
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std