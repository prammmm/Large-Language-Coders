from __future__ import annotations
from pathlib import Path
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from ber.utils.runtime import LOG
SOURCE_COLUMNS = ['entity_id', 'business_name', 'business_address', 'country']
GT_COLUMNS = ['source1_entity_id', 'matched_entity_ids']

def count_data_lines(path: Path, chunk: int=1 << 24) -> int:
    """Number of data lines (physical lines minus header), streaming."""
    n = 0
    last = b''
    with open(path, 'rb') as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            n += buf.count(b'\n')
            last = buf[-1:]
    if last and last != b'\n':
        n += 1
    return max(n - 1, 0)

def open_tsv(path: Path, columns: list[str], block_size: int=1 << 24):
    read_opts = pacsv.ReadOptions(block_size=block_size, use_threads=True)
    parse_opts = pacsv.ParseOptions(delimiter='\t', quote_char='"', double_quote=True, newlines_in_values=False)
    conv_opts = pacsv.ConvertOptions(column_types={c: pa.string() for c in columns}, strings_can_be_null=False, quoted_strings_can_be_null=False)
    return pacsv.open_csv(path, read_options=read_opts, parse_options=parse_opts, convert_options=conv_opts)

def ingest_file(src: Path, dst: Path, columns: list[str], block_size: int=1 << 24) -> dict:
    """Stream ``src`` TSV into ``dst`` Parquet. Returns stats; raises on any mismatch."""
    reader = open_tsv(src, columns, block_size)
    got = reader.schema.names
    if got != columns:
        raise ValueError(f'{src}: header {got} != expected {columns}')
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix('.parquet.tmp')
    rows = 0
    schema = pa.schema([(c, pa.string()) for c in columns])
    with pq.ParquetWriter(tmp, schema, compression='zstd') as w:
        for batch in reader:
            batch = pa.RecordBatch.from_arrays([batch.column(c).cast(pa.string()) for c in columns], schema=schema)
            w.write_batch(batch)
            rows += batch.num_rows
    expected = count_data_lines(src)
    if rows != expected:
        tmp.unlink(missing_ok=True)
        raise ValueError(f'{src}: parsed {rows} rows but file has {expected} data lines')
    tmp.replace(dst)
    LOG.info('ingested %s -> %s (%d rows)', src, dst, rows)
    return {'src': str(src), 'dst': str(dst), 'rows': rows}

def ingest_split(paths, split: str, block_size: int=1 << 24, force: bool=False) -> list[dict]:
    """Ingest source1/2/3 (+ ground truth for train if present)."""
    stats = []
    jobs = [(s, SOURCE_COLUMNS) for s in ('source1', 'source2', 'source3')]
    if paths.raw_file(split, 'ground_truth').exists():
        jobs.append(('ground_truth', GT_COLUMNS))
    for source, cols in jobs:
        src, dst = (paths.raw_file(split, source), paths.interim_file(split, source))
        if dst.exists() and (not force):
            LOG.info('skip %s (exists; use --force to rebuild)', dst)
            stats.append({'src': str(src), 'dst': str(dst), 'rows': pq.ParquetFile(dst).metadata.num_rows, 'skipped': True})
            continue
        stats.append(ingest_file(src, dst, cols, block_size))
    return stats
