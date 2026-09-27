"""Frozen, deterministic S1-level validation split (DEC-004).

Stratified by (country_key, match-count bucket); within each stratum the S1 ids are
sorted and a seeded permutation picks round(val_fraction * size) of them. The result is
written once to ``data/splits/val_s1_ids.txt`` and is never silently regenerated.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ber.evaluation.metrics import match_bucket
from ber.utils.runtime import LOG, write_json


def match_counts(matched_col) -> np.ndarray:
    """Number of ids in each comma-separated list ('' -> 0)."""
    lists = pc.split_pattern(matched_col, ",")
    lens = pc.list_value_length(lists).to_numpy(zero_copy_only=False).astype(np.int64)
    empty = pc.equal(pc.utf8_trim_whitespace(matched_col), "").to_numpy(zero_copy_only=False)
    lens[empty] = 0
    return lens


def make_split(paths, val_fraction: float, seed: int, force: bool = False) -> dict:
    out = paths.val_ids
    if out.exists() and not force:
        raise FileExistsError(f"{out} already exists; the split is frozen. Pass --force only if the team agreed to regenerate it.")
    gt = pq.read_table(paths.interim_file("train", "ground_truth"))
    s1 = pq.read_table(paths.interim_file("train", "source1"), columns=["entity_id", "country"])
    from ber.data.normalize_records import country_key

    country_of = dict(zip(s1.column("entity_id").to_pylist(), (country_key(c) for c in s1.column("country").to_pylist())))
    ids = gt.column("source1_entity_id").to_pylist()
    counts = match_counts(gt.column("matched_entity_ids"))
    missing = [i for i in ids if i not in country_of]
    if missing:
        raise ValueError(f"{len(missing)} ground-truth S1 ids are not in train_source1, e.g. {missing[:3]}")

    strata: dict[str, list[str]] = {}
    for sid, n in zip(ids, counts):
        strata.setdefault(f"{country_of[sid]}|{match_bucket(int(n))}", []).append(sid)

    rng = np.random.default_rng(seed)
    val: list[str] = []
    per_stratum = {}
    for key in sorted(strata):
        members = sorted(strata[key])
        take = int(round(val_fraction * len(members)))
        perm = rng.permutation(len(members))[:take]
        val.extend(members[i] for i in perm)
        per_stratum[key] = {"size": len(members), "val": take}
    val.sort()

    out.parent.mkdir(parents=True, exist_ok=True)
    payload = ("\n".join(val) + "\n").encode("utf-8")
    out.write_bytes(payload)
    meta = {
        "seed": seed,
        "val_fraction": val_fraction,
        "n_train_s1": len(ids),
        "n_val_s1": len(val),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "strata": per_stratum,
    }
    write_json(out.with_name("val_split_meta.json"), meta)
    LOG.info("validation split: %d / %d S1 (sha256 %s)", len(val), len(ids), meta["sha256"][:12])
    return meta


def load_val_ids(paths) -> list[str]:
    if not paths.val_ids.exists():
        raise FileNotFoundError(f"{paths.val_ids} missing: run the `split` stage first")
    return [line.strip() for line in paths.val_ids.read_text(encoding="utf-8").splitlines() if line.strip()]
