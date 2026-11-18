# 1.5s atom window straddles two of the annotation's 1s caption bins,
# so by default it is scored on the words from both (--caption-mode span). 
# do singl row as he sensitivity check(will drop like a third of the film's words)

#python3 nndb2_stimuli_normative.py --offset -5 --n-perm 4999
# python3 nndb2_stimuli_normative.py --offset -5 --caption-mode single --tag caption_single

import argparse
import csv
import json

import os
import sys

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage, optimal_leaf_ordering
from scipy.spatial.distance import squareform

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here) # nndb2_stimuli_utils pulls the root module in from there

import nndb2_stimuli_utils as su


def ring_order(fmats, names):
    # spoke order for the coin figure. seriate the norms by how they actually
    # covary across TRs so neighbours behave alike, then rotate the weakest join
    # round to the wrap point. deliberately built from the stimulus side only:
    # seriating on the atom z-profiles would arrange the spokes to manufacture
    # the arcs, and it goes stale every time an analysis choice moves
    pooled = np.vstack([fmats[m] for m in sorted(fmats)])
    corr = np.corrcoef(pooled, rowvar=False)
    d = np.clip(1 - corr, 0, 2)
    np.fill_diagonal(d, 0)
    cond = squareform(d, checks=False)
    order = list(leaves_list(optimal_leaf_ordering(linkage(cond, 'average'), cond)))
    adj = [corr[order[i], order[(i + 1) % len(order)]] for i in range(len(order))]
    cut = int(np.argmin(adj))
    order = order[cut + 1:] + order[:cut + 1]
    smooth = float(np.mean([corr[order[i], order[(i + 1) % len(order)]]
                            for i in range(len(order))]))
    return [names[i] for i in order], corr, smooth


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--offset', type=int, default=0,
                    help='shift atoms against embedding rows, in seconds, negative is roughly the HRF lag')
    ap.add_argument('--matrix', default='semantic_matrix.npy')
    ap.add_argument('--features', nargs='+', default=None, help='subset of norm dimensions')
    ap.add_argument('--text-source', choices=['all', 'dialogue', 'visual', 'env'], default='all')
    ap.add_argument('--caption-mode', choices=['span', 'single'], default='span',
                    help='rows a 1.5 s window is scored on: both bins it touches, or only the one it opens on')
    ap.add_argument('--require-text', action='store_true',
                    help='only keep TRs that actually have words, and z-score over those')
    ap.add_argument('--n-perm', type=int, default=4999)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default=None, help='write to a subdir instead of clobbering the main run')
    args = ap.parse_args()

    su.check_inputs()
    outdir = su.ensure_dir(su.normative_dir / args.tag if args.tag else su.normative_dir)
    rng = np.random.default_rng(args.seed)
    legend = su.atom_names()

    print(f'loading norms from {su.norm_manifest}')
    names, dicts = su.load_norms(su.norm_manifest, args.features)
    print(f'  {len(names)} dims: {", ".join(names)}')

    print('building the aligned stack...')
    stack = su.build_stack(offset=args.offset, matrix=args.matrix,
                           caption_mode=args.caption_mode)

    print(f'scoring TRs (text={args.text_source}, captions={args.caption_mode}, require_text={args.require_text})')
    fmats, coverage, analysis_stack = {}, {}, {}
    for movie, mv in stack.items():
        f, ntok = su.build_norm_features(movie, mv['cap_rows'], names, dicts, args.text_source)
        if args.require_text:
            keep = ntok > 0
            f = f[keep]
            onehots = [oh[keep] for oh in mv['onehots']]
        else:
            onehots = mv['onehots']
        fmats[movie] = su.zscore_within_movie(f)
        analysis_stack[movie] = {'onehots': onehots, 'emb_rows': mv['emb_rows']}
        coverage[movie] = (float((ntok > 0).mean()), int((ntok > 0).sum()),
                           float(ntok.mean()), len(ntok))
        print(f'  {movie:24s} {coverage[movie][0]:6.1%} of TRs have a word '
              f'({coverage[movie][1]:,} of {coverage[movie][3]:,})')

    with open(outdir / 'text_coverage.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['movie', 'n_trs_used', 'n_trs_with_words', 'frac_with_words', 'mean_tokens'])
        for movie in sorted(coverage):
            frac, n_words, mean_tok, n_used = coverage[movie]
            w.writerow([movie, n_used, n_words, round(frac, 4), round(mean_tok, 2)])

    print(f'circular-shift test, {args.n_perm} perms')
    z, q, means = su.run_feature_test(analysis_stack, fmats, names, args.n_perm, rng,
                                      outdir, legend, prefix='norm ')

    # top norms per atom, signed, strongest first. only the ones that survive FDR
    top = {}
    for a in su.atom_ids:
        top[legend[a]] = [(names[j], round(float(z[a - 1, j]), 2), round(float(means[a - 1, j]), 4))
                          for j in np.argsort(-np.abs(z[a - 1])) if q[a - 1, j] < 0.05][:8]

    ring, norm_corr, smooth = ring_order(fmats, names)
    print(f'ring order seriated, neighbour smoothness {smooth:.3f}')
    with open(outdir / 'norm_correlations.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow([''] + names)
        for i, nm in enumerate(names):
            w.writerow([nm] + [round(float(norm_corr[i, j]), 4) for j in range(len(names))])

    n_occ = sum(len(mv['emb_rows']) * mv['n_part'] for mv in stack.values())
    (outdir / 'summary.json').write_text(json.dumps({
        'analysis': 'atom x lexical norms',
        'offset': args.offset,
        'matrix': args.matrix,
        'text_source': args.text_source,
        'caption_mode': args.caption_mode,
        'scan_tr': su.scan_tr,
        'annotation_tr': su.annotation_tr,
        'require_text': args.require_text,
        'n_perm': args.n_perm,
        'seed': args.seed,
        'manifest': str(su.norm_manifest),
        'norm_dimensions': names,
        'n_participants': len(su.participant_movie_map()),
        'n_films': len(stack),
        'n_occurrences': n_occ,
        'n_cells': int(q.size),
        'n_sig_cells_q05': int((q < 0.05).sum()),
        'max_abs_effect_sd': round(float(np.abs(means).max()), 4),
        'top_norms_per_atom': top,
        'norm_ring_order': ring,
        'norm_ring_smoothness': round(smooth, 4),
    }, indent=2))

    n_sig = int((q < 0.05).sum())
    n_atoms_sig = sum(1 for a in su.atom_ids if (q[a - 1] < 0.05).any())
    print(f'\n{n_sig}/{q.size} atom x norm cells survive FDR, over {n_atoms_sig}/{su.n_atoms} atoms')
    print(f'biggest effect anywhere is {np.abs(means).max():.3f} SD')
    print(f'wrote it all toooooo {outdir}')
    print('\nYour time has come, run to the hills')


if __name__ == '__main__':
    main()
