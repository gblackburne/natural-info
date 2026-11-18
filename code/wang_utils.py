
import os
import numpy as np
import pandas as pd
import nibabel as nib

_this_dir = os.path.dirname(os.path.abspath(__file__))
_atlas_dir = os.path.join(_this_dir, '..', 'fMRI_general', 'atlases')
_sch_nii = os.path.join(_atlas_dir, 'schaefer_extend_atlases', 'extended_schaefer_1000.nii.gz')
_sch_csv = os.path.join(_atlas_dir, 'schaefer_extend_atlases', 'extended_schaefer_1000.csv')
_wang_dir = os.path.join(_atlas_dir, 'vtpm_atlases', 'ProbAtlas_v4', 'subj_vol_all')
_cache = os.path.join(_this_dir, 'results', 'main', 'extended_schaefer_1000', 'wang_overlap_counts.npy')

n_regions = 1054

# Wang ROIfiles_Labeling.txt order, label i sits at index i-1
wang_areas = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'VO1', 'VO2',
              'PHC1', 'PHC2', 'MST', 'hMT', 'LO2', 'LO1', 'V3b', 'V3a', 'IPS0',
              'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'SPL1', 'FEF']

_coarse = {'V1v': 'V1', 'V1d': 'V1', 'V2v': 'V2', 'V2d': 'V2',
           'V3v': 'V3', 'V3d': 'V3', 'hV4': 'V4',
           'IPS0': 'IPS/SPL', 'IPS1': 'IPS/SPL', 'IPS2': 'IPS/SPL',
           'IPS3': 'IPS/SPL', 'IPS4': 'IPS/SPL', 'IPS5': 'IPS/SPL',
           'SPL1': 'IPS/SPL'}


def coarse_area(area):
    """Map a native Wang area to its merged name.

    Collapses the ventral and dorsal halves of V1-V3, and pools the six
    intraparietal fields with SPL1, which carry no reliable ordering.
    """
    return _coarse.get(area, area)


wang_areas_merged = list(dict.fromkeys(coarse_area(a) for a in wang_areas))

def compute_overlap(cache=True):
    """
    Voxel-overlap count matrix between Schaefer parcels and Wang areas.

    Returns an (n_regions, 25) int array where entry [p, a] is the number of
    voxels of Schaefer label p+1 that fall inside Wang area a+1. Cached to disk
    after the first call.
    """
    if cache and os.path.exists(_cache):
        return np.load(_cache)

    sch = np.asarray(nib.load(_sch_nii).dataobj).astype(np.int32)
    lh = np.asarray(nib.load(os.path.join(_wang_dir, 'maxprob_vol_lh.nii.gz')).dataobj).astype(np.int32)
    rh = np.asarray(nib.load(os.path.join(_wang_dir, 'maxprob_vol_rh.nii.gz')).dataobj).astype(np.int32)

    # the two hemispheres never label the same voxel, so a plain sum combines them
    wang = np.where(lh > 0, lh, rh)

    mask = (sch > 0) & (wang > 0)
    s = sch[mask] - 1
    w = wang[mask] - 1
    counts = np.zeros((n_regions, len(wang_areas)), dtype=np.int64)
    np.add.at(counts, (s, w), 1)

    if cache:
        os.makedirs(os.path.dirname(_cache), exist_ok=True)
        np.save(_cache, counts)
    return counts


def _merge_columns(counts):
    """Collapse an (n_regions, 25) matrix onto the merged area set by summing."""
    out = np.zeros((counts.shape[0], len(wang_areas_merged)), dtype=counts.dtype)
    for j, a in enumerate(wang_areas):
        out[:, wang_areas_merged.index(coarse_area(a))] += counts[:, j]
    return out


def aggregate_to_wang(values, merge_vd=False, min_area_vox=1):
    """
    Overlap-weighted aggregation of a region vector into per-area values.

    Parameters
    ----------
    values : array, shape (n_regions,)
        Per-region result (any of the group_*_regions.npy vectors). NaNs are
        dropped from the weighting.
    merge_vd : bool
        False -> 25 native Wang areas; True -> 16 merged areas (V1-V3
        ventral/dorsal collapsed, IPS0-5 and SPL1 pooled as IPS/SPL).
    min_area_vox : int
        Areas whose total overlapping voxel count (over non-NaN parcels) is
        below this are returned as NaN rather than a fragile estimate.

    Returns
    -------
    pandas.Series indexed by area name, value = voxel-weighted mean of `values`.
    """
    values = np.asarray(values, dtype=float).ravel()
    if values.shape[0] != n_regions:
        raise ValueError(f'expected {n_regions} regions, got {values.shape[0]}')

    counts = compute_overlap()
    if merge_vd:
        counts = _merge_columns(counts)
        names = wang_areas_merged
    else:
        names = wang_areas

    finite = np.isfinite(values)
    w = counts.copy().astype(float)
    w[~finite, :] = 0.0
    v = np.where(finite, values, 0.0)

    tot = w.sum(axis=0)
    num = (w * v[:, None]).sum(axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        agg = num / tot
    agg[tot < min_area_vox] = np.nan
    return pd.Series(agg, index=names, name='value')


def wang_composition(min_frac=0.0):
    """
    Per-parcel breakdown of Wang overlap, for the parcel->area mapping table.

    Returns a DataFrame (one row per Schaefer parcel that overlaps any Wang
    area) with the parcel identity, its voxel count, the fraction of the parcel
    covered by retinotopic cortex, the dominant area (split and merged), and the
    top-3 area composition as a readable string.
    """
    counts = compute_overlap()
    df = pd.read_csv(_sch_csv)

    # parcel sizes from the label volume
    sch = np.asarray(nib.load(_sch_nii).dataobj).astype(np.int32)
    sizes = np.bincount(sch.ravel(), minlength=n_regions + 1)[1:n_regions + 1]

    rows = []
    for i in range(n_regions):
        c = counts[i]
        if c.sum() == 0:
            continue
        cov = c.sum() / sizes[i] if sizes[i] else 0.0
        top = int(np.argmax(c))
        order = np.argsort(c)[::-1]
        comp = ', '.join(f'{wang_areas[j]}:{c[j] / sizes[i]:.0%}'
                         for j in order[:3] if c[j] > 0)
        rows.append({
            'region': df.iloc[i]['region'],
            'label': int(df.iloc[i]['label']),
            'hemisphere': df.iloc[i]['hemisphere'],
            'network': df.iloc[i]['network'],
            'nvox': int(sizes[i]),
            'wang_coverage': cov,
            'wang_area': wang_areas[top],
            'wang_area_merged': coarse_area(wang_areas[top]),
            'top_frac': c[top] / sizes[i],
            'composition': comp,
        })
    out = pd.DataFrame(rows)
    return out[out['wang_coverage'] >= min_frac].reset_index(drop=True)
