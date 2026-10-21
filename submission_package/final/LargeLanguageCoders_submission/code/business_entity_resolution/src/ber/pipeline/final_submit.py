from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import yaml
from ber.config import Paths, load_config
from ber.decision.competition import best_claimant_keep
from ber.models import model_path
from ber.pipeline import run_stage2 as S
from ber.pipeline.run_baseline import _ids, pool_ids
from ber.submission.writer import CANDIDATE_HEADER, MATCHING_HEADER, write_id_lists
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging
SHIFT = S.SHIFT

def member(x, ref_sorted):
    if len(ref_sorted) == 0:
        return np.zeros(len(x), bool)
    pos = np.minimum(np.searchsorted(ref_sorted, x), len(ref_sorted) - 1)
    return ref_sorted[pos] == x

def stage1_accept(paths, tag, t):
    s1s, pools, scs = ([], [], [])
    for p in sorted(paths.scored_dir('test', tag).glob('*.parquet')):
        tb = pq.read_table(p, columns=['s1_row', 'pool_row', 'score'])
        sc = tb.column('score').to_numpy().astype(np.float32)
        m = sc >= np.float32(t)
        s1s.append(tb.column('s1_row').to_numpy().astype(np.int64)[m])
        pools.append(tb.column('pool_row').to_numpy().astype(np.int64)[m])
        scs.append(sc[m])
    s1, pool, sc = (np.concatenate(s1s), np.concatenate(pools), np.concatenate(scs))
    acc = best_claimant_keep(pool, s1, sc, np.ones(len(s1), bool))
    return np.sort(s1[acc] * SHIFT + pool[acc])

def vetoed(cfg_file):
    cfg = load_config(cfg_file)
    paths = Paths(cfg)
    ctl = stage1_accept(paths, cfg['scoring']['tag'], S.stage1_threshold(cfg))
    booster, meta = S.load_stage2(cfg, paths)
    s1, pool, prob, sc = S.predict_all(cfg, paths, 'test', booster, meta['features'])
    fin = S.final_decisions(cfg, s1, pool, prob, sc, meta, ctl)
    return np.sort(s1[fin] * SHIFT + pool[fin])

def candidates(paths, tag):
    s1s, pools, rks = ([], [], [])
    for p in sorted(paths.scored_dir('test', tag).glob('*.parquet')):
        tb = pq.read_table(p, columns=['s1_row', 'pool_row', 'cheap_rank'])
        s1s.append(tb.column('s1_row').to_numpy().astype(np.int64))
        pools.append(tb.column('pool_row').to_numpy().astype(np.int64))
        rks.append(tb.column('cheap_rank').to_numpy().astype(np.int64))
    return (np.concatenate(s1s), np.concatenate(pools), np.concatenate(rks))

def add_unmatched(base, extra):
    extra = extra[~member(extra, base)]
    extra = extra[~member(extra % SHIFT, np.unique(base % SHIFT))]
    return np.sort(np.concatenate([base, extra]))

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='configs/final.yaml')
    a = ap.parse_args(argv)
    setup_logging()
    configure_arrow()
    fc = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    c1, c2, c3 = (load_config(fc[k]) for k in ('primary', 'secondary', 'tertiary'))
    paths = Paths(c1)
    out = Path(fc['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    with PeakMemory() as pm:
        primary = vetoed(fc['primary'])
        secondary = vetoed(fc['secondary'])
        t3 = float(json.loads(model_path(c3, Paths(c3)).read_text(encoding='utf-8'))['meta']['threshold'])
        tertiary = stage1_accept(paths, c3['scoring']['tag'], t3)
        lists = [candidates(paths, c['scoring']['tag']) for c in (c1, c2, c3)]
        codes = [s * SHIFT + p for s, p, _ in lists]
        seen1, seen2 = (np.sort(codes[0]), np.sort(codes[1]))
        final = add_unmatched(primary, secondary[~member(secondary, seen1)])
        final = add_unmatched(final, tertiary[~member(tertiary, seen1) & ~member(tertiary, seen2)])
        seen = seen1
        parts = []
        for g, ((s, p, r), cd) in enumerate(zip(lists, codes)):
            keep = np.ones(len(cd), bool) if g == 0 else ~member(cd, seen)
            parts.append((s[keep], p[keep], np.full(int(keep.sum()), g, np.int64), r[keep]))
            if g:
                seen = np.sort(np.concatenate([seen, cd[keep]]))
        cs1 = np.concatenate([x[0] for x in parts])
        cpool = np.concatenate([x[1] for x in parts])
        o = np.lexsort((np.concatenate([x[3] for x in parts]), np.concatenate([x[2] for x in parts]), cs1))
        cs1, cpool = (cs1[o], cpool[o])
        del lists, codes, parts, seen, o
        code = cs1 * SHIFT + cpool
        if len(np.unique(code)) != len(code):
            raise RuntimeError('duplicate candidate')
        match = member(code, final)
        if int(match.sum()) != len(final):
            raise RuntimeError('match outside the candidate set')
        s1_ids = _ids(paths, 'test', 'source1').to_pylist()
        ids = pool_ids(paths, 'test')
        write_id_lists(out / 'candidate_pairs.tsv', CANDIDATE_HEADER, s1_ids, cs1, (cpool, ids))
        write_id_lists(out / 'matching_results.tsv', MATCHING_HEADER, s1_ids, cs1[match], (cpool[match], ids))
    LOG.info('final: %d matches, %d candidates -> %s | peak %.2f GB', len(final), len(code), out.as_posix(), pm.peak_gb)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
