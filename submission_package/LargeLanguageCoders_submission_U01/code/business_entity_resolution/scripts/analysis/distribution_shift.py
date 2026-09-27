"""Train-vs-test record-level distribution shift per source and country (READ-ONLY, streamed).

Per split x source x country: rows, name token count, name script class shares, empty
normalized name core, empty address, address has a number, address useful-word count, and the
pool size per S1 (candidate density driver).

    python scripts/analysis/distribution_shift.py --config configs/exp_b04_blocking_v2.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ber.config import Paths, load_config
from ber.data.normalize_records import load_manifest
from ber.utils.runtime import LOG, configure_arrow, setup_logging, write_json

COLS = ["name_core", "name_script", "addr_norm", "addr_nums", "addr_words"]


def profile(f: Path) -> dict:
    n = 0
    acc = {"name_tokens": 0.0, "name_core_empty": 0, "addr_empty": 0, "addr_has_number": 0, "addr_words": 0.0,
           "script_latin": 0, "script_mixed": 0, "script_nonlatin": 0, "script_noletters": 0}
    for b in pq.ParquetFile(f).iter_batches(columns=COLS, batch_size=500_000):
        n += b.num_rows
        nc = b.column("name_core")
        acc["name_tokens"] += float(pc.sum(pc.list_value_length(pc.utf8_split_whitespace(nc))).as_py() or 0)
        acc["name_core_empty"] += int(pc.sum(pc.equal(nc, "")).as_py() or 0)
        acc["addr_empty"] += int(pc.sum(pc.equal(b.column("addr_norm"), "")).as_py() or 0)
        acc["addr_has_number"] += int(pc.sum(pc.not_equal(b.column("addr_nums"), "")).as_py() or 0)
        acc["addr_words"] += float(pc.sum(pc.list_value_length(pc.utf8_split_whitespace(b.column("addr_words")))).as_py() or 0)
        s = b.column("name_script").to_numpy(zero_copy_only=False)
        for i, k in enumerate(("script_latin", "script_mixed", "script_nonlatin", "script_noletters")):
            acc[k] += int((s == i).sum())
    return {"rows": n} | {k: (v / n if n else None) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/exp_b04_blocking_v2.yaml")
    ap.add_argument("--out", default="experiments/runs/POST-0.91-FORENSIC/distribution_shift.json")
    a = ap.parse_args()
    setup_logging()
    configure_arrow()
    paths = Paths(load_config(a.config))
    rep = {}
    for split in ("train", "test"):
        man = load_manifest(paths, split)
        for source in ("source1", "source2", "source3"):
            for key, meta in sorted(man[source]["countries"].items()):
                r = profile(paths.processed_dir(split) / source / meta["file"])
                rep[f"{split}|{source}|{key}"] = r
                LOG.info("%-5s %-7s %-7s rows %9d tok %.2f empty_addr %.3f num %.3f words %.2f latin %.3f nonlatin %.3f mixed %.3f",
                         split, source, key, r["rows"], r["name_tokens"], r["addr_empty"], r["addr_has_number"], r["addr_words"],
                         r["script_latin"], r["script_nonlatin"], r["script_mixed"])
        for key in man["source1"]["countries"]:
            s1 = man["source1"]["countries"][key]["rows"]
            pool = sum(man[s]["countries"].get(key, {}).get("rows", 0) for s in ("source2", "source3"))
            rep[f"{split}|pool_per_s1|{key}"] = pool / s1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_json(Path(a.out), rep)
    for k, v in rep.items():
        if "pool_per_s1" in k:
            LOG.info("%s %.3f", k, v)


if __name__ == "__main__":
    main()
