"""
Edited copy of Temporal Interval Network Density Analysis (TINDA).
see <https://github.com/OHBA-analysis/osl-dynamics/blob/main/osl_dynamics/analysis/tinda.py>

Originally from osl-dynamics, licensed under the Apache License 2.0
(https://github.com/OHBA-analysis/osl-dynamics/blob/main/LICENSE).
Modified for this project by George Blackburne.
"""

import logging
from itertools import permutations

import numpy as np
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import interp1d

_logger = logging.getLogger("log-beep")


def find_intervals(tc_hot):
    """
    Find intervals (periods where :code:`tc_hot` is zero) in a hot vector.

    Parameters
    ----------
    tc_hot : array_like
        Hot vector (i.e., binary vector) of shape (n_samples,) or
        (n_samples, 1). For example, a hot vector of a state time course
        of shape (n_samples, n_states).

    Returns
    -------
    intervals : list
        List of tuples of start and end indices of intervals.
    durations : array_like
        Array of durations of intervals (in samples).
    """
    # Ensure tc_hot is 1D
    tc_hot = np.squeeze(tc_hot)
    
    # Handle edge case: all zeros (state never occurs)
    if np.all(tc_hot == 0):
        return [], np.array([])
    
    # Handle edge case: all ones (state always active, no intervals between)
    if np.all(tc_hot == 1):
        return [], np.array([])
    
    # Add padding to handle edge cases
    tc_tmp = np.concatenate([[0], tc_hot, [0]])
    
    # Find transitions
    diff_tc = np.diff(tc_tmp)
    start_indices = np.where(diff_tc == 1)[0]  # 0->1 transitions (start of active periods)
    end_indices = np.where(diff_tc == -1)[0]   # 1->0 transitions (end of active periods)
    
    # Find intervals BETWEEN active periods (where tc_hot is zero)
    intervals = []
    if len(start_indices) > 0 and len(end_indices) > 0:
        # Ensure we have the same number of starts and ends, or handle gracefully
        if len(start_indices) == len(end_indices):
            # Equal starts and ends - intervals are between consecutive active periods
            for i in range(len(end_indices) - 1):
                if end_indices[i] < start_indices[i + 1]:
                    intervals.append((end_indices[i], start_indices[i + 1]))
        else:
            # Unequal starts and ends - handle edge cases
            min_len = min(len(start_indices), len(end_indices))
            for i in range(min_len - 1):
                if i < len(end_indices) and (i + 1) < len(start_indices):
                    if end_indices[i] < start_indices[i + 1]:
                        intervals.append((end_indices[i], start_indices[i + 1]))
    
    # Compute durations safely
    if len(intervals) > 0:
        durations = np.array([end - start for start, end in intervals])
    else:
        durations = np.array([])
    
    return intervals, durations

def split_intervals(intervals, n_bins=2):
    """
    Splits each interval into nbin equally sized bins.

    Parameters
    ----------
    intervals : list
        List of tuples of start and end indices of intervals.
    n_bins : int, optional
        Number of bins to split each interval into.

    Returns
    -------
    divided_intervals : list
        List the same length as intervals (minus dropped intervals, see below),
        with each element being a list of tuples of start and end indices of
        bins.
    bin_sizes : list
        List of bin sizes (in samples), one per interval.
    drop_mask : array_like
        Array of zeros and ones indicating whether the interval was dropped
        because it was smaller than :code:`n_bins`.
    """
    divided_intervals = []
    bin_sizes = []
    drop_mask = np.zeros(len(intervals))
    
    for i, (interval_start, interval_end) in enumerate(intervals):
        n_samples = interval_end - interval_start
        if n_samples < n_bins:
            drop_mask[i] = 1
            continue

        bin_size = n_samples // n_bins
        remainder = n_samples % n_bins
        bins = []
        bin_start = interval_start
        for j in range(n_bins):
            bin_end = bin_start + bin_size
            bins.append((bin_start, bin_end))
            if remainder > 0:
                if remainder == n_bins - 1:
                    bin_end += 1
                elif j == (n_bins // 2 - 1):
                    bin_end += remainder
            bin_start = bin_end
        divided_intervals.append(bins)
        bin_sizes.append(bin_size)

    return divided_intervals, bin_sizes, drop_mask

def split_interval_duration(
    durations, interval_range=None, mode="sample", sampling_frequency=None
):
    """Split interval durations into bins based on their duration.

    Parameters
    ----------
    durations : array_like
        Array of durations of intervals (in samples).
    interval_range : array_like, optional
        Array of bin edges (in samples, seconds, or percentiles) to split
        durations into bins are defined as :code:`[>=interval_range[i],
        <interval_range[i+1])`. If :code:`None`, all durations are in the
        same bin.
    mode : str, optional
        Mode of interval_range, either :code:`"sample"` (e.g.,
        :code:`[4, 20, 100]`), :code:`"perc"` (e.g., :code:`range(20,100,20)`),
        or :code:`"sec"` (e.g., :code:`[0, 0.01, 0.1, 1, np.inf]`). If
        :code:`"sec"`, :code:`sfreq` must be provided.
    sampling_frequency : float, optional
        Sampling frequency (in Hz) of the data, only used if :code:`mode` is
        :code:`"sec"`.

    Returns
    -------
    mask : list
        List of arrays of zeros and ones indicating whether the interval was
        in the bin.
    interval_range : array_like
        Array of bin edges (in samples) used to split durations into bins.
    """
    # Handle empty durations array
    if len(durations) == 0:
        mask = [np.array([], dtype=int)]
        if interval_range is None:
            interval_range = [0, 0]  # Default range for empty case
        return mask, interval_range
    
    if interval_range is None:
        mask = [np.ones_like(durations, dtype=int)]
        interval_range = [np.min(durations), np.max(durations)]
    else:
        if mode == "sec" and sampling_frequency is None:
            raise ValueError(
                "Sampling frequency (sfreq) must be specified when mode is 'sec'"
            )
        
        if mode == "sec":
            interval_range = [r * sampling_frequency for r in interval_range]
        elif mode == "perc":
            # Handle empty durations for percentile calculation
            if len(durations) == 0:
                interval_range = [0] * len(interval_range)
            else:
                interval_range = np.percentile(durations, interval_range)
        
        mask = []
        for i, start in enumerate(interval_range):
            if i < len(interval_range) - 1:
                hot_vector = np.logical_and(
                    durations >= start, durations < interval_range[i + 1]
                )
                mask.append(hot_vector.astype(int))
    
    return mask, interval_range

def compute_fo_stats(
    tc_sec, divided_intervals, interval_mask=None, return_all_intervals=False
):
    """Compute sums and weighted averages of time courses in each interval.

    Parameters
    ----------
    tc_sec : array_like
        Time course of shape (n_samples, n_states).
    divided_intervals : list
        List with each element corresponding to an interval, each itself being a
        list of tuples of start and end indices of interval bins.
    interval_mask : array_like, optional
        Array of zeros and ones indicating whether the interval was in the bin.
    return_all_intervals : bool, optional
        Whether to return the density/sum of all intervals in addition to the
        interval averages/sums.

    Returns
    -------
    interval_weighted_avg : array_like
        Array of weighted averages of time courses in each interval of shape
        (n_states, n_bins, n_interval_ranges).
    interval_sum : array_like
        Array of sums of time courses in each interval of shape (n_states,
        n_bins, n_interval_ranges).
    interval_weighted_avg_all : list
        List of length n_interval_ranges with each element an array of weighted
        averages of time courses in each interval of shape (n_states, n_bins,
        n_intervals). :code:`None` if :code:`return_all_intervals=False`
        (default).
    interval_sum_all : list
        List of length :code:`n_interval_ranges` with each element an array of
        sums of time courses in each interval of shape (n_states, n_bins,
        n_intervals). :code:`None` if :code:`return_all_intervals=False`.
    """
    if interval_mask is None:
        interval_mask = [np.ones(len(divided_intervals))]

    # Handle empty divided_intervals
    if len(divided_intervals) == 0:
        n_bins = 2  # default
        n_mask = len(interval_mask)
        interval_weighted_avg = np.zeros((tc_sec.shape[1], n_bins, n_mask))
        interval_sum = np.zeros((tc_sec.shape[1], n_bins, n_mask))
        if return_all_intervals:
            return interval_weighted_avg, interval_sum, [], []
        else:
            return interval_weighted_avg, interval_sum, None, None

    if (
        len(divided_intervals[0]) == 2
    ):  # this corresponds to the matlab code but only works for two bins
        # and I think it's less principled than the code below
        intervals = []
        for interval in divided_intervals:
            intervals.append([interval[0][0], interval[-1][-1]])

        interval_weighted_avg = np.zeros((tc_sec.shape[1], 2, len(interval_mask)))
        interval_sum = np.zeros((tc_sec.shape[1], 2, len(interval_mask)))
        temp_to = []
        temp_away = []
        for i in intervals:
            d = int(np.floor((np.diff(i) - 1) / 2))
            temp_away.append(tc_sec[i[0] : i[0] + d + 1, :])
            temp_to.append(tc_sec[i[1] - d - 1 : i[1], :])

        interval_weighted_avg_all = []
        interval_sum_all = []
        for j, mask in enumerate(interval_mask):
            mask_indices = np.where(mask == 1)[0]
            if len(mask_indices) > 0:
                temp_to_flat = np.concatenate([temp_to[k] for k in mask_indices])
                temp_away_flat = np.concatenate([temp_away[k] for k in mask_indices])

                interval_weighted_avg[:, 0, j] = np.mean(temp_away_flat, axis=0)
                interval_weighted_avg[:, 1, j] = np.mean(temp_to_flat, axis=0)
                interval_sum[:, 0, j] = np.sum(temp_away_flat, axis=0)
                interval_sum[:, 1, j] = np.sum(temp_to_flat, axis=0)
                interval_weighted_avg_all.append(
                    np.transpose(
                        np.stack(
                            [
                                np.stack(
                                    [
                                        temp_away[k].mean(axis=0)
                                        for k in mask_indices
                                    ],
                                    axis=-1,
                                ),
                                np.stack(
                                    [
                                        temp_to[k].mean(axis=0)
                                        for k in mask_indices
                                    ],
                                    axis=-1,
                                ),
                            ]
                        ),
                        axes=[1, 0, 2],
                    )
                )
                interval_sum_all.append(
                    np.transpose(
                        np.stack(
                            [
                                np.stack(
                                    [
                                        temp_away[k].sum(axis=0)
                                        for k in mask_indices
                                    ],
                                    axis=-1,
                                ),
                                np.stack(
                                    [
                                        temp_to[k].sum(axis=0)
                                        for k in mask_indices
                                    ],
                                    axis=-1,
                                ),
                            ]
                        ),
                        axes=[1, 0, 2],
                    )
                )
            else:
                # Handle empty mask case
                interval_weighted_avg[:, :, j] = 0
                interval_sum[:, :, j] = 0
                interval_weighted_avg_all.append(np.zeros((tc_sec.shape[1], 2, 0)))
                interval_sum_all.append(np.zeros((tc_sec.shape[1], 2, 0)))
    else:
        # TODO: I think this is more principled than the matlab code,
        # but I need to check
        interval_sum = np.zeros(
            (tc_sec.shape[1], len(divided_intervals[0]), len(divided_intervals))
        )
        interval_weighted_avg = np.zeros(
            (tc_sec.shape[1], len(divided_intervals[0]), len(divided_intervals))
        )
        for i, interval in enumerate(divided_intervals):
            for j, (start, end) in enumerate(interval):
                interval_sum[:, j, i] = np.sum(tc_sec[start:end, :], axis=0)
            # Fix potential division by zero
            bin_length = end - start
            if bin_length > 0:
                interval_weighted_avg[:, :, i] = interval_sum[:, :, i] / bin_length
            else:
                interval_weighted_avg[:, :, i] = 0

        interval_weighted_avg_all = [
            interval_weighted_avg[:, :, interval_selection == 1]
            for interval_selection in interval_mask
        ]
        interval_sum_all = [
            interval_sum[:, :, interval_selection == 1]
            for interval_selection in interval_mask
        ]
        
        # Handle empty selections
        interval_weighted_avg_list = []
        interval_sum_list = []
        for weighted_avg in interval_weighted_avg_all:
            if weighted_avg.shape[-1] > 0:
                interval_weighted_avg_list.append(weighted_avg.mean(axis=-1))
            else:
                interval_weighted_avg_list.append(np.zeros(weighted_avg.shape[:-1]))
        
        for int_sum in interval_sum_all:
            if int_sum.shape[-1] > 0:
                interval_sum_list.append(int_sum.mean(axis=-1))
            else:
                interval_sum_list.append(np.zeros(int_sum.shape[:-1]))
        
        interval_weighted_avg = np.stack(interval_weighted_avg_list, axis=-1)
        interval_sum = np.stack(interval_sum_list, axis=-1)

    if return_all_intervals:
        return (
            interval_weighted_avg,
            interval_sum,
            interval_weighted_avg_all,
            interval_sum_all,
        )
    else:
        return interval_weighted_avg, interval_sum, None, None

def collate_stats(stats, field, all_to_all=False, ignore_elements=None):
    """Collate list of stats (e.g., of different states) into a single array.

    Parameters
    ----------
    stats : list
        List of stats (:code:`dict`) for each state. Each element is a
        dictionary with keys that at least should include "field" (e.g.,
        :code:`interval_wavg`), that is the output of :code:`compute_fo_stats`.
    field : str
        Field of stats to collate, e.g., :code:`"interval_wavg"`,
        :code:`"interval_sum"`.
    all_to_all : bool, optional
        Whether the density_of was used to compute the stats (in which case
        the first 2 dimensions are not :code:`n_states` x :code:`n_states`).
        Default is :code:`False`.
    ignore_elements : list, optional
        List of indices in stats to ignore (i.e. because they don't contain
        binary events).

    Returns
    -------
    collated_stat : array_like
        The collated stat (:code:`n_interval_states`, :code:`n_density_states`,
        :code:`n_bins`, :code:`n_interval_ranges`). If :code:`all_to_all=False`
        (default) (i.e., when the density is computed for all states using all
        states' intervals), then the first two dimensions are :code:`n_states`
        and the diagonal is :code:`np.nan`.
    """
    if ignore_elements is None:
        ignore_elements = []

    num_states = len(stats)
    # Find first non-None stat to get shape
    valid_stats = [s for s in stats if s is not None]
    if not valid_stats:
        # All stats are None, return appropriate empty array
        return np.array([])
    
    shp = valid_stats[0][field].shape  # (n_states, n_bins, n_interval_ranges)
    
    if num_states > 1:
        if all_to_all:
            collated_stat = np.full((2 * [num_states] + list(shp[1:])), np.nan)
        else:
            collated_stat = np.full(([num_states] + list(shp)), np.nan)

        for i in range(num_states):
            if i in ignore_elements or stats[i] is None:
                # intervals are not binary, keep a row of nans
                continue
            if all_to_all:
                collated_stat[i, np.arange(num_states) != i, ...] = stats[i][field]
            else:
                collated_stat[i] = stats[i][field]

    else:
        if stats[0] is not None:
            collated_stat = stats[0][field]
        else:
            collated_stat = np.array([])

    return collated_stat

def tinda(
    tc,
    density_of=None,
    n_bins=2,
    interval_mode=None,
    interval_range=None,
    sampling_frequency=None,
    return_all_intervals=False,
):
    """Compute time-in-state density and sum for each interval.

    Parameters
    ----------
    tc : array_like
        Time courses of shape (n_samples, n_states) define intervals from will
        use the same time courses to compute density of when :code:`density_of`
        is :code:`None`. Can be a list of time courses (e.g. state time courses
        for each individual).
    density_of : array_like, optional
        Time course of shape (n_samples, n_states) to compute density of if
        :code:`None` (default), density is computed for all columns of tc.
    n_bins : int, optional
        Number of bins to divide each interval into (default 2).
    interval_mode : str, optional
        Mode of :code:`interval_range`, either :code:`"sample"` (default),
        "sec" (seconds) or "perc" (percentile). To interpret the interval range
        as seconds, :code:`sfreq` must be provided.
    interval_range : array_like, optional
        Array of bin edges (in samples, seconds, or percentiles) used to split
        durations into bins (default :code:`None`), e.g.
        :code:`np.arange(0, 1, 0.1)` for 100 ms bins.
    sampling_frequency : float, optional
        Sampling frequency of tc (in Hz), only used if
        :code:`interval_mode="sec"`.
    return_all_intervals : bool, optional
        Whether to return the density/sum of all intervals in addition to the
        interval averages/sums. If :code:`True`, will return a list of arrays in
        :code:`stats[i]['all_interval_wavg'/'all_interval_sum']`, each
        corresponding to an interval range.

    Returns
    -------
    fo_density : array_like
        Time-in-state densities array of shape (n_interval_states,
        n_density_states, n_bins, n_interval_ranges). :code:`n_interval_states`
        is the number of states in the interval time courses (i.e., tc);
        :code:`n_density_states` is the number of states in the density time
        courses (i.e., :code:`density_of`). If :code:`density_of` is
        :code:`None`, :code:`n_density_states` is the same as
        :code:`n_interval_states`. If tc is a list of time courses (e.g., state
        time courses for multiple individuals), then an extra dimension is appended
        for the individuals.
    fo_sum : array_like
        Same as :code:`fo_density`, but with time-in-state sums instead of
        densities.
    stats : dict
        Dictionary of stats, including

        - :code:`durations`: interval durations in samples.
        - :code:`intervals`: start/end samples for each interval (intervals).
        - :code:`interval_wavg`: the weighted average (i.e, time-in-state
          density) over all interval.
        - :code:`interval_sum`: the sum (i.e., time-in-state) over all
          intervals.
        - :code:`divided_intervals`: the bin edges for each interval.
        - :code:`bin_sizes`: the bin sizes for each interval.
        - :code:`interval_range`: the interval range (in samples).
        - :code:`all_interval_wavg`: unaveraged interval densities (only if
          :code:`return_all_intervals=True`).
        - :code:`all_interval_sum`: unaveraged interval sums (only if
          :code:`return_all_intervals=True`).
    """
    if isinstance(
        tc, list
    ):  # list of time courses (e.g., individuals' HMM state time courses)
        if density_of is None:
            fo_density_tmp, fo_sum_tmp, stats = zip(
                *[
                    tinda(
                        itc,
                        None,
                        n_bins,
                        interval_mode,
                        interval_range,
                        sampling_frequency,
                        return_all_intervals,
                    )
                    for itc in tc
                ]
            )
        elif len(density_of) == len(tc):
            fo_density_tmp, fo_sum_tmp, stats = zip(
                *[
                    tinda(
                        itc,
                        density_of[ix],
                        n_bins,
                        interval_mode,
                        interval_range,
                        sampling_frequency,
                        return_all_intervals,
                    )
                    for ix, itc in enumerate(tc)
                ]
            )
        fo_density = np.stack(fo_density_tmp, axis=-1)
        fo_sum = np.stack(fo_sum_tmp, axis=-1)

    else:
        stats = []
        dim = tc.shape
        ignore_elements = []

        for i in range(dim[1]):
            itc_prim = tc[:, i]
            if np.all(itc_prim == 0):
                _logger.info(f"Skipping state {i}: no activations detected.")

                stats.append(
                    {
                        "durations": np.array([]),
                        "intervals": [],
                        "interval_wavg": np.full((tc.shape[1] - 1, n_bins, 1), 0),
                        "interval_sum": np.full((tc.shape[1] - 1, n_bins, 1), 0),
                        "divided_intervals": [],
                        "bin_sizes": [],
                        "interval_range": [0, 0],
                        "all_interval_wavg": [],
                        "all_interval_sum": [],
                    }
                )
                continue

            if not np.array_equal(
                itc_prim, itc_prim.astype(int)
            ):  # if not binary (i.e., intervals are not well defined)
                stats.append(None)
                ignore_elements.append(i)

            else:
                if density_of is None:
                    # we're doing density of all states in all states' intervals
                    itc_sec = tc[:, np.setdiff1d(range(dim[1]), i)]
                else:
                    itc_sec = density_of

                # get interval info
                intervals, durations = find_intervals(itc_prim)
                divided_intervals, bin_sizes, dropped_intervals = split_intervals(
                    intervals, n_bins
                )  # split intervals into nbin
                durations = durations[
                    dropped_intervals == 0
                ]  # drop intervals that are too short to be split into nbin
                interval_mask, interval_range_samples = split_interval_duration(
                    durations,
                    interval_range=interval_range,
                    mode=interval_mode,
                    sampling_frequency=sampling_frequency,
                )  # split intervals into interval_range (i.e.,
                # to compute statistics of intervals with durations in a
                # certain range)

                # Compute time-in-state densities and sums in all intervals
                (
                    interval_wavg,
                    interval_sum,
                    all_interval_wavg,
                    all_interval_sum,
                ) = compute_fo_stats(
                    itc_sec,
                    divided_intervals,
                    interval_mask,
                    return_all_intervals=return_all_intervals,
                )

                # Append stats
                stats.append(
                    {
                        "durations": durations,
                        "intervals": intervals,
                        "interval_wavg": interval_wavg,
                        "interval_sum": interval_sum,
                        "divided_intervals": divided_intervals,
                        "bin_sizes": bin_sizes,
                        "interval_range": interval_range_samples,
                        "all_interval_wavg": all_interval_wavg,
                        "all_interval_sum": all_interval_sum,
                    }
                )

        # Get a full matrix of FO densities and sums
        fo_density = collate_stats(
            stats,
            "interval_wavg",
            all_to_all=density_of is None,
            ignore_elements=ignore_elements,
        )
        fo_sum = collate_stats(
            stats,
            "interval_sum",
            all_to_all=density_of is None,
            ignore_elements=ignore_elements,
        )

    return fo_density, fo_sum, stats

def get_mean_direction(fo_density):
    # pos = post, neg = pre ref state dominance
    mean_direction = np.squeeze(
        np.nanmean((fo_density[:, :, 0] - fo_density[:, :, 1]), axis=-1)
    )
    mean_direction[np.isnan(mean_direction)] = 0
    return mean_direction


def optimise_sequence(fo_density, metric_to_use=0, n_perms=10**6):
    """Optimise the sequence to maximal circularity.

    This function reads in the mean pattern of differential fractional
    occupancy and computes the optimal display for a sequential circular plot
    visualization.

    Parameters
    ----------
    fo_density : array_like
        Time-in-state densities array of shape (n_interval_states,
        n_density_states, 2, n_sessions).
    metric : int, optional
        Metric to use for optimisation:

        - :code:`0`: mean FO asymmetry.
        - :code:`1`: proportional FO asymmetry (i.e. asymmetry as a proportion
          of a baseline - which time spend in the state).
        - :code:`2`: proportional FO asymmetry using global baseline FO, rather
          than a individual-specific baseline.

    Returns
    -------
    best_sequence : list
        List of best sequence of states to plot (in order of counterclockwise
        rotation).
    """
    if len(fo_density.shape) == 5:
        fo_density = np.squeeze(fo_density)

    # make sure there are no nans:
    fo_density[np.isnan(fo_density)] = 0

    # Compute different metrics to optimise
    metric = []
    metric.append(np.mean(fo_density[:, :, 0, :] - fo_density[:, :, 1, :], axis=2))
    temp = (fo_density[:, :, 0, :] - fo_density[:, :, 1, :]) / np.mean(
        fo_density, axis=2
    )
    temp[np.isnan(temp)] = 0
    metric.append(np.mean(temp, axis=2))
    metric.append(
        np.mean(fo_density[:, :, 0, :] - fo_density[:, :, 1, :], axis=2)
        / np.mean(fo_density, axis=(2, 3))
    )
    n_metrics = len(metric)
    K = fo_density.shape[0]

    best_sequence = []
    for i in range(n_metrics):
        ix = np.arange(K)
        v = np.imag(np.sum(circle_angles(ix) * metric[i]))
        cnt = 0
        while cnt < n_perms:
            cnt += 1
            swaps = np.random.permutation(K)
            swaps = swaps[:2]
            tmpix = ix.copy()
            tmpix[swaps[0]] = ix[swaps[1]]
            tmpix[swaps[1]] = ix[swaps[0]]
            tmpv = np.imag(np.sum(circle_angles(tmpix) * metric[i]))
            if tmpv < v:
                v = tmpv
                ix = tmpix
        best_sequence.append(np.roll(ix, -np.where([iix == 0 for iix in ix])[0][0]))
    # Return the best sequence for the chosen metric (in order of counterclockwise
    # rotation)
    return best_sequence[metric_to_use]

def optimise_sequence_stream(mean_direction, n_perms=10**6):
    # streamlined optim
    K = mean_direction.shape[0]
    ix = np.arange(K)
    v = np.imag(np.sum(circle_angles(ix) * mean_direction))
    cnt = 0
    while cnt < n_perms:
        cnt += 1
        swaps = np.random.permutation(K)[:2]
        tmpix = ix.copy()
        tmpix[swaps[0]] = ix[swaps[1]]
        tmpix[swaps[1]] = ix[swaps[0]]
        tmpv = np.imag(np.sum(circle_angles(tmpix) * mean_direction))
        if tmpv < v:
            v = tmpv
            ix = tmpix
    best_sequence = np.roll(ix, -np.where(ix == 0)[0][0])
    return best_sequence

def circle_angles(order):
    """Compute the phase differences between states in a circular plot.

    Parameters
    ----------
    order : list
        List of state orders (in order of counterclockwise
        rotation).

    Returns
    -------
    angleplot : array_like
        Array of phase differences between states in a circular plot.

    """
    K = len(order)
    disttoplot_manual = np.zeros(K, dtype=complex)
    for i3 in range(K):
        disttoplot_manual[order[i3]] = np.exp(1j * (i3 + 1) / K * 2 * np.pi)

    angleplot = np.exp(
        1j
        * (
            np.angle(disttoplot_manual[:, np.newaxis]).T
            - np.angle(disttoplot_manual[:, np.newaxis])
        )
    )
    return angleplot

def compute_cycle_strength(angleplot, asym, relative=True, whichstate=None):
    if len(asym.shape) == 3:
        tmp = np.stack(
            [angleplot * asym[:, :, i] for i in range(asym.shape[2])], axis=-1
        )
    else:
        tmp = angleplot * asym
    if whichstate is not None:
        # Note that we are counting each (i,j) double because for the rotational
        # momentum per state we take into account (i,j) and (j,i) for all j and one
        # particular i.
        tmp = np.squeeze(
            tmp[
                whichstate,
                :,
            ]
        ) + np.squeeze(tmp[:, whichstate])
        cycle_strength = np.imag(np.nansum(tmp, axis=0))
    else:
        cycle_strength = np.imag(np.nansum(tmp, axis=(0, 1)))

    # positive rotational momentum should indicate clockwise cycle
    cycle_strength = -cycle_strength

    if relative:  # normalise by the theoretical maximum
        cycle_strength = cycle_strength / np.abs(
            compute_cycle_strength(
                angleplot,
                np.sign(np.imag(angleplot)),
                relative=False,
                whichstate=whichstate,
            )
        )

    return cycle_strength

# modified plotting funcs:

def interpolate_circular_colors(colors, n_segments=2000):
    """Interpolate colors around a circle with smooth transitions.
    
    Parameters
    ----------
    colors : array_like
        Array of RGB colors at discrete positions.
    n_segments : int
        Number of segments for smooth interpolation.
    
    Returns
    -------
    interpolated_colors : array
        Array of interpolated RGB colors.
    """
    K = len(colors)
    
    original_positions = np.linspace(0, 1, K, endpoint=False)
    new_positions = np.linspace(0, 1, n_segments, endpoint=False)
    interpolated_colors = np.zeros((n_segments, 3))
    
    for channel in range(3):
        channel_values = colors[:, channel]
        extended_positions = np.concatenate([original_positions - 1, original_positions, original_positions + 1])
        extended_values = np.tile(channel_values, 3)
        interp_func = interp1d(extended_positions, extended_values, kind='cubic')
        interpolated_colors[:, channel] = interp_func(new_positions)
    
    # Ensure values are in valid range [0, 1]
    interpolated_colors = np.clip(interpolated_colors, 0, 1)
    
    return interpolated_colors

def plot_cycle(ordering, av_direction, edges, new_figure=False, color_scheme=None,
               state_names=None, band=True, band_alpha=0.7, node_radius=0.11,
               node_halo=True, halo_scale=1.18, label=True, cmap_cyc=None):
    """Plot the state cycle as a circular diagram.

    Each node is a coloured inner circle (``color_scheme[state]``) with a black
    border, sitting on a slightly larger white circle; arrows are drawn for the
    nonzero entries of ``edges``, their direction following the sign of
    ``av_direction``.

    Parameters
    ----------
    ordering : list
        State indices in counterclockwise order around the circle.
    av_direction : array_like
        (K, K) mean FO asymmetry; the sign of entry (j, k) sets each arrow's direction.
    edges : array_like
        (K, K) array; a nonzero entry (j, k) draws an arrow between states j and k.
    new_figure : bool, optional
        Create a new figure (default False).
    color_scheme : array_like, optional
        (K, 3) RGB colours (or list of matplotlib colours) indexed by state, used to
        fill the nodes and the surrounding band. If None, a default palette is used.
    state_names : dict or list, optional
        Node labels. A dict may be keyed by 0-based state index or 1-based state
        number; a list is indexed by state. If None, the 1-based state number is used.
    band : bool, optional
        Draw the coloured ring behind the nodes (default True).
    band_alpha : float, optional
        Transparency of the ring (default 0.7).
    node_radius : float, optional
        Radius of the coloured inner circle (default 0.11).
    node_halo : bool, optional
        Draw the white circle underneath each node (default True).
    halo_scale : float, optional
        Radius of the white circle relative to node_radius (default 1.18).
    label : bool, optional
        Write the label inside each node (default True).
    cmap_cyc : colormap, optional
        Fallback gradient colormap for the band when color_scheme is None.
    """
    from matplotlib.colors import to_rgb

    default_scheme = np.array([
        [0, 0, 1.0], [1.0, 0.3333, 0], [1.0, 0.6667, 0], [0.6667, 1.0, 0.3333],
        [0.3333, 1.0, 0.6667], [0, 1.0, 1.0], [0.5529, 0.8275, 0.7804],
        [1.0, 0.5, 0.5], [0, 0.6667, 1.0], [1.0, 1.0, 0], [0.7451, 0.7294, 0.8549],
        [0.6667, 0, 0],
    ])
    K = len(ordering)
    has_colors = color_scheme is not None
    if has_colors:
        color_scheme = np.array([to_rgb(c) if isinstance(c, str) else c[:3]
                                 for c in color_scheme])
    else:
        color_scheme = default_scheme[:K]

    def node_label(state):
        # dict: keyed by 1-based state number (as in `legend`); list: 0-based index
        if isinstance(state_names, dict):
            if (state + 1) in state_names:
                return str(state_names[state + 1])
            if state in state_names:
                return str(state_names[state])
        elif state_names is not None:
            return str(state_names[state])
        return str(int(state) + 1)

    fig = plt.figure(figsize=(8, 8)) if np.any(new_figure) else plt.gcf()
    ax = plt.axes([0, 0, 1, 1])
    ax.set_aspect('equal')

    # reorder everything into the clockwise display order
    ordering = np.roll(np.asarray(ordering)[::-1], 1)
    edges = edges[ordering][:, ordering]
    mean_direction = av_direction[ordering][:, ordering]
    node_rgb = color_scheme[ordering]

    theta = np.arange(0, 2 * np.pi, 2 * np.pi / K)
    x = np.roll(np.cos(theta), int(K / 4))
    y = np.roll(np.sin(theta), int(K - (K / 4)))
    pos = np.stack([x, y]).T

    inner_radius, outer_radius, n_segments = 0.85, 1.15, 2000

    # coloured band behind the nodes
    if band:
        if has_colors:
            node_angles = np.arctan2(pos[:, 1], pos[:, 0])
            node_angles = np.where(node_angles < 0, node_angles + 2 * np.pi, node_angles)
            sorted_colors = node_rgb[np.argsort(node_angles)]
            band_colors = interpolate_circular_colors(sorted_colors, n_segments)
        else:
            if cmap_cyc is None:
                import cmasher as cmr
                cmap_cyc = cmr.copper_s
            band_colors = np.array([cmap_cyc(i / n_segments)[:3] for i in range(n_segments)])
        angles = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
        wedge_angle = 2 * np.pi / n_segments
        for i, angle in enumerate(angles):
            ax.add_patch(Wedge((0, 0), outer_radius, np.degrees(angle),
                         np.degrees(angle + wedge_angle), width=outer_radius - inner_radius,
                         facecolor=band_colors[i], edgecolor='none', zorder=-5,
                         alpha=band_alpha))

    # arrows for the requested edges
    for j in range(K):
        for k in range(K):
            if not edges[j, k]:
                continue
            d = pos[k] - pos[j]
            d = d / np.sqrt(np.sum(d ** 2))
            start, end = pos[j] + 0.15 * d, pos[k] - 0.15 * d
            s, e = (start, end) if mean_direction[j, k] > 0 else (end, start)
            ax.add_patch(FancyArrowPatch(s, e, connectionstyle='arc3,rad=-0.1',
                         arrowstyle='-|>', mutation_scale=20, linewidth=2,
                         color='black', alpha=0.9, zorder=1))

    # nodes: white circle underneath, coloured inner circle with black border on top
    for i in range(K):
        if node_halo:
            ax.add_patch(plt.Circle(pos[i], node_radius * halo_scale, facecolor='white',
                         zorder=2, edgecolor='black', linewidth=1))
        ax.add_patch(plt.Circle(pos[i], node_radius, facecolor=node_rgb[i],
                     edgecolor='black', linewidth=1, zorder=3))
        if label:
            lum = float(np.dot(np.asarray(node_rgb[i])[:3], [0.299, 0.587, 0.114]))
            ax.text(pos[i, 0], pos[i, 1], node_label(int(ordering[i])), ha='center',
                    va='center', color='white' if lum < 0.5 else 'black',
                    fontsize=12, fontweight='bold', zorder=4)

    ax.axis('off')
    ax.set_frame_on(False)
    ax.set(xticks=[], yticks=[])
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    return fig, ax
