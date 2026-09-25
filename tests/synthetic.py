"""Deterministic synthetic mini-universe in the official file format (for tests only).

Train covers US + India; test adds France (open-set country). Files are written with
CRLF line endings and one RFC-4180 quoted field, like the real data.
"""

from __future__ import annotations

import random
from pathlib import Path

import yaml

FIRST = ["Summit", "Apex", "Harbor", "Lotus", "Crescent", "Maple", "Orion", "Nova", "Cedar", "Falcon", "Indigo", "Zenith"]
SECOND = ["Partners", "Traders", "Clinic", "Bakery", "Motors", "Textiles", "Studio", "Foods", "Logistics", "Pharma"]
SUFFIX = {"us": ["Inc", "LLC", "Corp"], "india": ["Pvt Ltd", "Private Limited", "LLP"], "france": ["SARL", "SAS", "SA"]}
STREETS = {
    "us": ["Oak Street", "Pine Avenue", "Lake Road", "Hill Drive", "Elm Lane"],
    "india": ["MG Road", "Gandhi Nagar", "Station Road", "Nehru Marg", "Temple Street"],
    "france": ["Rue de la Paix", "Avenue Victor Hugo", "Boulevard Voltaire", "Rue Lafayette"],
}
CITIES = {"us": ["Springfield, IL", "Dayton, OH", "Tulsa, OK"], "india": ["Pune, MH", "Chennai, TN", "Indore, MP"], "france": ["Lyon", "Nantes", "Lille"]}
COUNTRY_LABEL = {"us": "US", "india": "India", "france": "France"}
ABBREV = {"Street": "St", "Avenue": "Ave", "Road": "Rd", "Drive": "Dr", "Lane": "Ln", "Boulevard": "Bd", "Rue": "R."}
DEVANAGARI = {"Traders": "ट्रेडर्स", "Pharma": "फार्मा", "Foods": "फूड्स", "Clinic": "क्लिनिक"}


def _typo(s: str, rng: random.Random) -> str:
    i = rng.randrange(1, max(2, len(s) - 1))
    return s[:i] + s[i + 1 :] if s[i] != " " else s


def _variant(name: str, addr: str, country: str, second: str, rng: random.Random) -> tuple[str, str]:
    kind = rng.randrange(5)
    n, a = name, addr
    if kind == 0:
        n, a = n.upper(), a.upper()
    elif kind == 1:
        for full, ab in ABBREV.items():
            a = a.replace(full, ab)
        n = n.rsplit(" ", 1)[0] if " " in n else n
    elif kind == 2:
        n = _typo(n, rng)
    elif kind == 3 and country == "india" and second in DEVANAGARI:
        n = n.split(" ")[0] + " " + DEVANAGARI[second]
    else:
        parts = [p.strip() for p in a.split(",")]
        a = ", ".join(parts[1:] + parts[:1])
    return n, a


def build_split(split: str, countries: list[str], n_s1: int, rng: random.Random, id_base: int):
    s1, s2, s3, gt = [], [], [], []
    nid = id_base
    for i in range(n_s1):
        c = countries[i % len(countries)]
        second = rng.choice(SECOND)
        base = f"{FIRST[i % len(FIRST)]} {second} {i}"
        name = f"{base} {rng.choice(SUFFIX[c])}"
        addr = f"{rng.randrange(1, 999)} {rng.choice(STREETS[c])}, {rng.choice(CITIES[c])}"
        sid = f"S1-{nid:06d}"
        nid += 1
        s1.append((sid, name, addr, COUNTRY_LABEL[c]))
        n_match = 0 if i % 7 == 0 else rng.randrange(1, 4)  # ~1/7 singletons
        matched = []
        for _ in range(n_match):
            vn, va = _variant(name, addr, c, second, rng)
            src = rng.choice((2, 3))
            mid = f"S{src}-{nid:06d}"
            nid += 1
            (s2 if src == 2 else s3).append((mid, vn, va, COUNTRY_LABEL[c]))
            matched.append(mid)
        gt.append((sid, ",".join(matched)))
    # distractors: unrelated businesses in the same countries
    for j in range(n_s1 // 2):
        c = countries[j % len(countries)]
        src = 2 if j % 2 else 3
        rec = (f"S{src}-{nid:06d}", f"Other {rng.choice(SECOND)} {j + 5000}", f"{rng.randrange(1000, 1999)} Far Away Street, Nowhere", COUNTRY_LABEL[c])
        nid += 1
        (s2 if src == 2 else s3).append(rec)
    # one RFC-quoted address + one empty address
    if s2:
        eid, n, a, c = s2[0]
        s2[0] = (eid, n, a + ', "C" Wing', c)
    if s3:
        eid, n, a, c = s3[-1]
        s3[-1] = (eid, n, "", c)
    return s1, s2, s3, gt, nid


def _write_tsv(path: Path, header: list[str], rows: list[tuple]) -> None:
    def fmt(v: str) -> str:
        return '"' + v.replace('"', '""') + '"' if '"' in v else v

    lines = ["\t".join(header)] + ["\t".join(fmt(v) for v in r) for r in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))


def make_synthetic_repo(root: Path, n_train: int = 140, n_test: int = 90) -> dict:
    rng = random.Random(42)
    cols = ["entity_id", "business_name", "business_address", "country"]
    tr = build_split("train", ["us", "india"], n_train, rng, 1)
    te = build_split("test", ["us", "india", "france"], n_test, rng, tr[4] + 1000)
    for split, (s1, s2, s3, gt, _) in (("train", tr), ("test", te)):
        d = root / "datasets" / split
        _write_tsv(d / f"{split}_source1.tsv", cols, s1)
        _write_tsv(d / f"{split}_source2.tsv", cols, s2)
        _write_tsv(d / f"{split}_source3.tsv", cols, s3)
        if split == "train":
            _write_tsv(d / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"], gt)
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "baseline_v0.yaml").read_text(encoding="utf-8"))
    cfg["paths"]["root"] = str(root)
    cfg["preprocess"]["n_workers"] = 1
    cfg["preprocess"]["batch_rows"] = 37  # force multiple batches
    cfg["blocking"]["max_raw_pairs_per_chunk"] = 200  # force multiple S1 chunks
    cfg["scoring"]["feature_batch_pairs"] = 50
    cfg["scoring"]["rapidfuzz_workers"] = 1
    cfg["split"]["val_fraction"] = 0.3
    cfg_path = root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return {"config": cfg_path, "train_gt": {s: g.split(",") if g else [] for s, g in tr[3]}, "test_s1": [r[0] for r in te[0]]}
