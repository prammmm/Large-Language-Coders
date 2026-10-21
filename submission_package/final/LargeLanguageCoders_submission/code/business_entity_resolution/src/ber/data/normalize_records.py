from __future__ import annotations
import hashlib
import re
from multiprocessing import get_context
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from ber.normalization.text import normalize_text
from ber.normalization.views import build_views
from ber.utils.runtime import LOG, read_json, write_json
PROCESSED_SCHEMA = pa.schema([('row', pa.int32()), ('entity_id', pa.string()), ('country_key', pa.string()), ('name_norm', pa.string()), ('name_core', pa.string()), ('name_concat', pa.string()), ('name_script', pa.int8()), ('addr_norm', pa.string()), ('addr_nums', pa.string()), ('addr_words', pa.string())])

def country_key(raw: str) -> str:
    return normalize_text(raw)

def country_file_name(key: str) -> str:
    slug = re.sub('[^a-z0-9]+', '_', key.encode('ascii', 'ignore').decode()).strip('_') or 'empty'
    digest = hashlib.sha1(key.encode('utf-8')).hexdigest()[:8]
    return f'{slug}_{digest}.parquet'

def _work(args):
    start, ids, names, addrs, countries = args
    views = build_views(names, addrs)
    views['country_key'] = [country_key(c) for c in countries]
    return (start, ids, views)

def normalize_source(src: Path, out_dir: Path, batch_rows: int, n_workers: int) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    pf = pq.ParquetFile(src)
    writers: dict[str, pq.ParquetWriter] = {}
    counts: dict[str, int] = {}
    files: dict[str, str] = {}

    def batches():
        start = 0
        for b in pf.iter_batches(batch_size=batch_rows, columns=['entity_id', 'business_name', 'business_address', 'country']):
            d = b.to_pydict()
            yield (start, d['entity_id'], d['business_name'], d['business_address'], d['country'])
            start += b.num_rows

    def consume(start, ids, views):
        n = len(ids)
        table = pa.table({'row': pa.array(range(start, start + n), pa.int32()), 'entity_id': pa.array(ids, pa.string()), **{c: pa.array(views[c], PROCESSED_SCHEMA.field(c).type) for c in PROCESSED_SCHEMA.names if c not in ('row', 'entity_id')}}, schema=PROCESSED_SCHEMA)
        keys = table.column('country_key').unique().to_pylist()
        for key in sorted(keys):
            part = table.filter(pc.equal(table.column('country_key'), key))
            if key not in writers:
                files[key] = country_file_name(key)
                writers[key] = pq.ParquetWriter(out_dir / files[key], PROCESSED_SCHEMA, compression='zstd')
                counts[key] = 0
            writers[key].write_table(part)
            counts[key] += part.num_rows
    total = 0
    try:
        if n_workers > 1:
            with get_context('spawn').Pool(n_workers) as pool:
                pending = []
                for job in batches():
                    pending.append(job)
                    if len(pending) == n_workers:
                        for res in pool.map(_work, pending):
                            consume(*res)
                            total += len(res[1])
                        pending = []
                if pending:
                    for res in pool.map(_work, pending):
                        consume(*res)
                        total += len(res[1])
        else:
            for job in batches():
                res = _work(job)
                consume(*res)
                total += len(res[1])
    finally:
        for w in writers.values():
            w.close()
    LOG.info('normalized %s: %d rows, countries=%s', src, total, counts)
    return {'rows': total, 'countries': {k: {'file': files[k], 'rows': counts[k]} for k in sorted(counts)}}

def normalize_split(paths, split: str, batch_rows: int, n_workers: int, force: bool=False) -> dict:
    out = paths.processed_dir(split)
    manifest_path = out / 'manifest.json'
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    for source in ('source1', 'source2', 'source3'):
        if source in manifest and (not force):
            LOG.info('skip normalize %s/%s (in manifest; use --force)', split, source)
            continue
        src_dir = out / source
        if src_dir.exists():
            for f in src_dir.glob('*.parquet'):
                f.unlink()
        manifest[source] = normalize_source(paths.interim_file(split, source), src_dir, batch_rows, n_workers)
        write_json(manifest_path, manifest)
    return manifest

def load_manifest(paths, split: str) -> dict:
    return read_json(paths.processed_dir(split) / 'manifest.json')

def read_country(paths, split: str, source: str, key: str, columns: list[str] | None=None) -> pa.Table:
    m = load_manifest(paths, split)[source]['countries']
    if key not in m:
        return PROCESSED_SCHEMA.empty_table() if columns is None else PROCESSED_SCHEMA.empty_table().select(columns)
    return pq.read_table(paths.processed_dir(split) / source / m[key]['file'], columns=columns)
