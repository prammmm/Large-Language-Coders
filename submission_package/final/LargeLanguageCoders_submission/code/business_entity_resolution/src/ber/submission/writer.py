from __future__ import annotations
from pathlib import Path
import numpy as np
import pyarrow as pa
MATCHING_HEADER = ('source1_entity_id', 'matched_entity_ids')
CANDIDATE_HEADER = ('source1_entity_id', 'candidate_entity_ids')
_FORBIDDEN = ('\t', ',', '\n', '\r', ' ', '"')

def write_id_lists(path: Path, header: tuple[str, str], s1_ids: list[str], pair_s1_row: np.ndarray, pair_ids: pa.Array | list[str] | tuple[np.ndarray, pa.Array], block_rows: int=50000) -> dict:
    """Write one row per ``s1_ids[r]``; the list for row r is ``pair_ids[j]`` for all j"""
    pair_s1_row = np.asarray(pair_s1_row, dtype=np.int64)
    if len(pair_s1_row) and np.any(np.diff(pair_s1_row) < 0):
        raise ValueError('pair_s1_row must be sorted ascending')
    if len(pair_s1_row) and (pair_s1_row[0] < 0 or pair_s1_row[-1] >= len(s1_ids)):
        raise ValueError('pair_s1_row out of range')
    if isinstance(pair_ids, list):
        pair_ids = pa.array(pair_ids, pa.string())
    if isinstance(pair_ids, tuple):
        idx, table = pair_ids
        idx = np.asarray(idx)

        def get_ids(lo, hi):
            return table.take(pa.array(idx[lo:hi])).to_pylist()
        n_pairs = len(idx)
    else:

        def get_ids(lo, hi):
            return pair_ids.slice(lo, hi - lo).to_pylist()
        n_pairs = len(pair_ids)
    if n_pairs != len(pair_s1_row):
        raise ValueError('pair_ids and pair_s1_row lengths differ')
    if len(set(s1_ids)) != len(s1_ids):
        raise ValueError('duplicate S1 ids passed to writer')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    n_rows = n_empty = n_ids = 0
    with open(tmp, 'wb') as f:
        f.write(('\t'.join(header) + '\n').encode('utf-8'))
        for a in range(0, len(s1_ids), block_rows):
            b = min(a + block_rows, len(s1_ids))
            lo, hi = np.searchsorted(pair_s1_row, [a, b])
            ids = get_ids(lo, hi)
            counts = np.bincount(pair_s1_row[lo:hi] - a, minlength=b - a)
            lines = []
            off = 0
            for r in range(a, b):
                c = int(counts[r - a])
                lst = ids[off:off + c]
                off += c
                s1 = s1_ids[r]
                if any((ch in s1 for ch in _FORBIDDEN)):
                    raise ValueError(f'illegal character in S1 id {s1!r}')
                if c:
                    if len(set(lst)) != c:
                        raise ValueError(f'duplicate ids in list for {s1}')
                    for x in lst:
                        if not x.startswith(('S2-', 'S3-')) or any((ch in x for ch in _FORBIDDEN)):
                            raise ValueError(f'illegal id {x!r} for {s1}')
                    lines.append(f"{s1}\t{','.join(lst)}")
                    n_ids += c
                else:
                    lines.append(f'{s1}\t')
                    n_empty += 1
            f.write(('\n'.join(lines) + '\n').encode('utf-8'))
            n_rows += b - a
    tmp.replace(path)
    return {'path': str(path), 'rows': n_rows, 'empty_rows': n_empty, 'ids': n_ids}
