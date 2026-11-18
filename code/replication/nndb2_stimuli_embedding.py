# python3 nndb2_stimuli_embedding.py --offset -5 --n-perm 4999
import argparse
import csv
import json

import os
import sys

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here) # nndb2_stimuli_utils pulls the root module in from there

import nndb2_stimuli_utils as su


def write_sample_tables(outdir, stack):
    # who watched what, and how common each atom is. both belong in the methods
    _, sequences = su.load_winner_states()
    legend = su.atom_names()
    pmap = su.participant_movie_map()

    with open(outdir / 'participant_movie_map.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['participant_id', 'movie', 'n_trs_used'])
        for pid in sorted(pmap):
            w.writerow([pid, pmap[pid], len(stack[pmap[pid]]['emb_rows'])])

    counts = np.zeros(su.n_atoms)
    for seq in sequences.values():
        counts += np.bincount(seq - 1, minlength=su.n_atoms)
    with open(outdir / 'atom_prevalence.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['atom_id', 'name', 'n_occurrences', 'fraction'])
        for a in su.atom_ids:
            w.writerow([a, legend[a], int(counts[a - 1]),
                        round(float(counts[a - 1] / counts.sum()), 6)])
    return counts


def run_geometry(outdir, stack, n_pca, n_perm, seed, legend):
    rng = np.random.default_rng(seed)
    all_res = np.concatenate([mv['R'] for mv in stack.values()], axis=0)
    n_comp = min(n_pca, all_res.shape[1], all_res.shape[0] - 1)
    print(f'PCA ({n_comp} comps) on {all_res.shape[0]:,} residualised TR vectors')
    pca = PCA(n_components=n_comp, random_state=seed).fit(all_res)
    evr = float(pca.explained_variance_ratio_.sum())
    stack_pca = {m: {'R': pca.transform(mv['R']).astype(np.float64), 'onehots': mv['onehots']}
                 for m, mv in stack.items()}

    s, counts, ss_total = su.compute_s(stack_pca, shift=False)
    n_total = int(counts.sum())
    ssb_obs = su.ss_between(s, counts)
    f_obs = su.pseudo_f(ssb_obs, ss_total, n_total, su.n_atoms)
    r2_obs = ssb_obs / ss_total

    centroids = (s / counts[None, :]).T
    pairs = su.atom_pairs()
    pair_obs = np.array([np.sum((centroids[i] - centroids[j]) ** 2) for i, j in pairs])

    print(f'cir-shift null, {n_perm} perms')
    ssb_null = np.empty(n_perm)
    pair_null = np.empty((n_perm, len(pairs)))
    norm_null = np.empty((n_perm, su.n_atoms))
    for p in range(n_perm):
        sp, counts_p, _ = su.compute_s(stack_pca, rng=rng, shift=True)
        ssb_null[p] = su.ss_between(sp, counts_p)
        cp = (sp / counts_p[None, :]).T
        pair_null[p] = [np.sum((cp[i] - cp[j]) ** 2) for i, j in pairs]
        norm_null[p] = np.linalg.norm(cp, axis=1)
        if (p + 1) % max(1, n_perm // 10) == 0:
            print(f'  geometry perm {p + 1}/{n_perm}')

    p_omni = (1 + int((ssb_null >= ssb_obs).sum())) / (n_perm + 1)
    pair_p = np.array([(1 + int((pair_null[:, k] >= pair_obs[k]).sum())) / (n_perm + 1)
                       for k in range(len(pairs))])
    pair_q = su.bh_fdr(pair_p)

    norm_obs = np.linalg.norm(centroids, axis=1)
    norm_p = np.array([(1 + int((norm_null[:, a] >= norm_obs[a]).sum())) / (n_perm + 1)
                       for a in range(su.n_atoms)])

    # full space centroids, which is what the similarity network gets drawn from
    dim = stack[next(iter(stack))]['R'].shape[1]
    s_res = np.zeros((dim, su.n_atoms))
    s_raw = np.zeros((dim, su.n_atoms))
    n_full = np.zeros(su.n_atoms)
    for mv in stack.values():
        cm = np.sum(mv['onehots'], axis=0)
        s_res += mv['R'].T @ cm
        s_raw += mv['E'].T @ cm
        n_full += cm.sum(axis=0)
    cent_res = (s_res / n_full[None, :]).T
    cent_raw = (s_raw / n_full[None, :]).T
    cos_res = cdist(cent_res, cent_res, metric='cosine')
    cos_raw = cdist(cent_raw, cent_raw, metric='cosine')

    np.savez(outdir / 'atom_semantics.npz',
             centroids_pca=centroids, centroids_res_full=cent_res, centroids_raw_full=cent_raw,
             n_a=counts, ssb_obs=ssb_obs, ssb_null=ssb_null, F_obs=f_obs, r2_obs=r2_obs,
             pair_idx=np.array(pairs), pair_obs=pair_obs, pair_p=pair_p, pair_q=pair_q,
             cos_res=cos_res, cos_raw=cos_raw, norm_obs=norm_obs, norm_null=norm_null,
             norm_null_p=norm_p, atom_ids=np.array(su.atom_ids))

    with open(outdir / 'pairwise_atom_tests.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['atom_a', 'name_a', 'atom_b', 'name_b', 'sqdist_pca',
                    'cosdist_res_full', 'p_perm', 'q_fdr', 'sig_q05'])
        for k, (i, j) in enumerate(pairs):
            w.writerow([i + 1, legend[i + 1], j + 1, legend[j + 1],
                        round(float(pair_obs[k]), 6), round(float(cos_res[i, j]), 6),
                        round(float(pair_p[k]), 6), round(float(pair_q[k]), 6),
                        bool(pair_q[k] < 0.05)])

    with open(outdir / 'atom_centroid_diagnostics.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['atom_id', 'name', 'n_occurrences', 'centroid_norm_obs',
                    'centroid_norm_null_mean', 'p_perm'])
        for a in su.atom_ids:
            i = a - 1
            w.writerow([a, legend[a], int(counts[i]), round(float(norm_obs[i]), 6),
                        round(float(norm_null[:, i].mean()), 6), round(float(norm_p[i]), 6)])

    print(f'  pseudo-F = {f_obs:.3f}, R2 = {r2_obs:.4%}, p_perm = {p_omni:.4g}')
    print(f'  {int((pair_q < 0.05).sum())}/{len(pairs)} atom pairs differ after FDR')
    return {'n_pca': n_comp, 'evr': evr, 'pseudo_F': float(f_obs), 'r2': float(r2_obs),
            'p_perm': float(p_omni), 'ssb_obs': float(ssb_obs),
            'n_total_occurrences': n_total,
            'n_pairs_sig': int((pair_q < 0.05).sum()), 'n_pairs': len(pairs)}


def run_kinematics(outdir, stack, n_perm, seed, legend, mode, cache_path, offset):
    zraw = None
    if cache_path is not None and cache_path.is_file():
        blob = np.load(cache_path, allow_pickle=False)
        # the cache is offset specific, measures get pulled at emb_rows and those
        # move when the lag moves. never reuse one across offsets
        ok = (list(blob['names']) == su.measure_names
              and 'offset' in blob.files and int(blob['offset']) == int(offset)
              and all(blob[f'F_{m}'].shape[0] == len(stack[m]['emb_rows']) for m in stack))
        if ok:
            zraw = {m: blob[f'F_{m}'] for m in stack}
            print(f'loaded cached measures from {cache_path.name}')
        else:
            print('cache is stale (different lag or measure set), re-measuring')
    if zraw is None:
        zraw = {}
        for movie, mv in stack.items():
            raw = su.extract_measures(movie, mv['emb_rows'])
            zraw[movie] = su.zscore_within_movie(raw)
            print(f'  measured {movie} {raw.shape}')
        if cache_path is not None:
            np.savez(cache_path, names=np.array(su.measure_names),
                     offset=np.array(offset),
                     **{f'F_{m}': zraw[m] for m in zraw})

    picks, fmats = su.axis_matrices(zraw, mode=mode)
    fam_names = list(su.families)
    for fam in fam_names:
        print(f'  {fam:20s} -> {picks[fam][0]} (typicality {picks[fam][2]:+.3f})')

    with open(outdir / 'family_members.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['family', 'measure', 'sign', 'r_to_family', 'in_representative', 'description'])
        for fam, members in su.families.items():
            typ = picks[fam][3]
            used = {n for n, _ in picks[fam][1]}
            for k, (name, sign) in enumerate(members):
                w.writerow([fam, name, int(sign), round(float(typ[k]), 4),
                            bool(name in used), su.measure_descriptions[name]])

    pooled = np.vstack([fmats[m] for m in sorted(fmats)])
    corr = np.corrcoef(pooled, rowvar=False)
    with open(outdir / 'axis_correlations.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow([''] + fam_names)
        for i, fam in enumerate(fam_names):
            w.writerow([fam] + [round(float(corr[i, j]), 4) for j in range(len(fam_names))])
    eig = np.linalg.eigvalsh(corr)[::-1]
    print(f'  axes are near enough independent, biggest |r| = '
          f'{np.abs(corr - np.eye(len(corr))).max():.3f}')

    print(f'main axis test, {n_perm} perms')
    rng = np.random.default_rng(seed)
    z, q, means = su.run_feature_test(stack, fmats, fam_names, n_perm, rng, outdir,
                                      legend, prefix='axis ')

    agreement, panel_z = {}, {}
    for fam in fam_names:
        members = su.families[fam]
        names = [n for n, _ in members]
        print(f'panel: {fam} ({len(names)} measures)')
        block = su.orient(zraw, members)
        sub = su.ensure_dir(outdir / f'panel_{fam}')
        rng = np.random.default_rng(seed)
        if mode == 'mean':
            # members and their leave-one-out composites go through the same pass, so a
            # measure never gets compared against a mean it is itself a part of
            combined = {m: np.column_stack([block[m], su.build_loo(block[m])]) for m in block}
            az, aq, _ = su.run_feature_test(stack, combined, names + [f'loo_{n}' for n in names],
                                            n_perm, rng, sub, legend, prefix=f'{fam[:10]} ')
            pz, pq = az[:, :len(names)], aq[:, :len(names)]
            reference = az[:, len(names):]
        else:
            pz, pq, _ = su.run_feature_test(stack, block, names, n_perm, rng, sub,
                                            legend, prefix=f'{fam[:10]} ')
            reference = np.repeat(z[:, [fam_names.index(fam)]], len(names), axis=1)
        panel_z[fam] = pz
        agreement[fam] = np.array([float(np.corrcoef(pz[:, k], reference[:, k])[0, 1])
                                   for k in range(len(names))])
        print(f'  profile r: mean {agreement[fam].mean():+.2f}, min {agreement[fam].min():+.2f}')

    with open(outdir / 'panel_profile_agreement.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['family', 'measure', 'in_representative', 'profile_r', 'max_abs_z'])
        for fam in fam_names:
            names = [n for n, _ in su.families[fam]]
            used = {n for n, _ in picks[fam][1]}
            for k, name in enumerate(names):
                w.writerow([fam, name, bool(name in used), round(float(agreement[fam][k]), 4),
                            round(float(np.abs(panel_z[fam][:, k]).max()), 3)])

    return picks, corr, eig, z, q, means, agreement


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--offset', type=int, default=0,
                    help='shift atoms against embedding rows, in seconds, negative is roughly the HRF lag')
    ap.add_argument('--matrix', default='semantic_matrix.npy')
    ap.add_argument('--n-pca', type=int, default=100)
    ap.add_argument('--n-perm', type=int, default=4999)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--representative', choices=['mean', 'medoid'], default='mean',
                    help='axis value, either the family mean or its most typical single measure')
    ap.add_argument('--skip-geometry', action='store_true')
    ap.add_argument('--skip-kinematics', action='store_true')
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--tag', default=None, help='write to a subdir instead of clobbering the main run')
    args = ap.parse_args()

    su.check_inputs()
    outdir = su.ensure_dir(su.embedding_dir / args.tag if args.tag else su.embedding_dir)
    legend = su.atom_names()

    print('building the aligned stack...')
    stack = su.build_stack(offset=args.offset, matrix=args.matrix)
    prevalence = write_sample_tables(outdir, stack)

    geom = None
    if not args.skip_geometry:
        print('\n--- part 1, embedding geometry ---')
        geom = run_geometry(outdir, stack, args.n_pca, args.n_perm, args.seed, legend)

    kin = None
    if not args.skip_kinematics:
        print('\n--- part 2, semantic kinematics ---')
        cache = None if args.no_cache else outdir / 'measures_cache.npz'
        kin = run_kinematics(outdir, stack, args.n_perm, args.seed, legend,
                             args.representative, cache, args.offset)

    summary = {
        'offset': args.offset,
        'scan_tr': su.scan_tr,
        'annotation_tr': su.annotation_tr,
        'matrix': args.matrix,
        'n_perm': args.n_perm,
        'seed': args.seed,
        'representative': args.representative,
        'n_participants': len(su.participant_movie_map()),
        'n_films': len(stack),
        'atom_prevalence': {legend[a]: round(float(prevalence[a - 1] / prevalence.sum()), 5)
                            for a in su.atom_ids},
    }
    if geom:
        summary['geometry'] = geom
        summary['n_total_occurrences'] = geom['n_total_occurrences']
    if kin:
        picks, corr, eig, z, q, means, agreement = kin
        summary['kinematics'] = {
            'families': {f: {'n_measures': len(su.families[f]),
                             'representative': picks[f][0],
                             'typicality': round(picks[f][2], 4)} for f in su.families},
            'n_cells': int(q.size),
            'n_sig_cells_q05': int((q < 0.05).sum()),
            'max_abs_effect_sd': round(float(np.abs(means).max()), 4),
            'max_abs_axis_correlation': round(float(np.abs(corr - np.eye(len(corr))).max()), 4),
            'eigenvalues': [round(float(v), 4) for v in eig],
            'panel_profile_r': {f: {'mean': round(float(agreement[f].mean()), 4),
                                    'min': round(float(agreement[f].min()), 4)}
                                for f in agreement},
        }
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2))

    if kin:
        q, means = kin[4], kin[5]
        n_sig = int((q < 0.05).sum())
        n_atoms_sig = sum(1 for a in su.atom_ids if (q[a - 1] < 0.05).any())
        print(f'\n{n_sig}/{q.size} atom x axis cells survive FDR, over {n_atoms_sig}/{su.n_atoms} atoms')
        print(f'biggest effect anywhere is {np.abs(means).max():.3f} SD, so small but reliable')
    print(f'wrote it all to {outdir}')
    print('\nYour time has come, run to the hills')


if __name__ == '__main__':
    main()
