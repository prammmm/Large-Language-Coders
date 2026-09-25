"""Read-only dataset profiling for the Business Entity Resolution challenge.

Phase 0/1 audit utility. It does NOT implement any matching logic; it only
measures properties of the provided data so that architecture decisions
(normalization, blocking, validation split) are grounded in facts.

Memory-conscious: uses pyarrow and processes one source file at a time.
Raw files are opened read-only and never modified.

Usage:
    python scripts/profiling/profile_datasets.py --data-dir datasets \
        --out reports/profiling/profile_summary.json --sample-s1 100000 --seed 42
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]

# Only used for *profiling probes* (share of tokens that are generic), not a
# production normalization list.
GENERIC_TOKENS = [
    "inc", "llc", "ltd", "limited", "pvt", "private", "corp", "corporation",
    "co", "company", "the", "and", "of", "sarl", "sas", "sa", "sci", "eurl",
    "groupe", "group", "llp", "plc", "public", "pte", "de", "la", "le", "et",
    "des", "du", "services", "service", "enterprises", "enterprise", "india",
    "trading", "solutions", "store", "shop",
]


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------
def read_tsv(path: Path, columns: list[str]) -> pa.Table:
    parse = pacsv.ParseOptions(delimiter="\t", quote_char='"', double_quote=True,
                               newlines_in_values=False)
    conv = pacsv.ConvertOptions(column_types={c: pa.string() for c in columns},
                                strings_can_be_null=False)
    read = pacsv.ReadOptions(block_size=64 << 20)
    return pacsv.read_csv(path, read_options=read, parse_options=parse, convert_options=conv)


def count_lines(path: Path) -> int:
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 << 20), b""):
            n += chunk.count(b"\n")
    return n


# --------------------------------------------------------------------------
# Text helpers (profiling-only normalisation)
# --------------------------------------------------------------------------
def norm_text(arr: pa.Array) -> pa.Array:
    """lowercase -> NFKD -> strip combining marks -> non-alnum to space -> squeeze."""
    a = pc.utf8_lower(arr)
    a = pc.utf8_normalize(a, form="NFKD")
    a = pc.replace_substring_regex(a, r"\p{Mn}+", "")
    a = pc.replace_substring_regex(a, r"[^\p{L}\p{N}]+", " ")
    return pc.utf8_trim_whitespace(a)


def tokens(arr: pa.Array) -> pa.Array:
    return pc.split_pattern(norm_text(arr), pattern=" ")


def rate(mask: pa.Array) -> float:
    n = len(mask)
    return round(float(pc.sum(pc.cast(mask, pa.int64())).as_py() or 0) / n, 6) if n else 0.0


def qtiles(arr: pa.Array, qs=(0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)) -> dict:
    arr = pc.cast(arr, pa.float64())
    vals = pc.quantile(arr, q=list(qs), interpolation="nearest").to_pylist()
    return {f"p{int(q * 100)}": v for q, v in zip(qs, vals)} | {"mean": round(pc.mean(arr).as_py(), 3)}


def hash_arr(arr) -> np.ndarray:
    """Process-local int64 hashes (memory-light set membership for overlap checks)."""
    return np.unique(np.fromiter((hash(x) for x in arr.to_pylist()), dtype=np.int64, count=len(arr)))


def top_counts(arr: pa.Array, k: int = 15) -> list:
    vc = pc.value_counts(arr)
    vals, cnts = vc.field("values").to_pylist(), vc.field("counts").to_numpy()
    order = np.argsort(-cnts)[:k]
    return [[vals[i], int(cnts[i])] for i in order]


PATTERNS = {
    "has_digit": r"\d",
    "non_ascii": r"[^\x00-\x7F]",
    "devanagari": r"\p{Devanagari}",
    "other_indic_script": r"[\p{Bengali}\p{Gujarati}\p{Gurmukhi}\p{Kannada}\p{Malayalam}\p{Oriya}\p{Tamil}\p{Telugu}]",
    "latin_accented": r"[À-ÿ]",
    "leading_junk": r"^[^\p{L}\p{N}]",
    "all_upper_latin": r"^[^a-z]*[A-Z][^a-z]*$",
    "literal_null_token": r"(?i)(^|[ ,])(null|nan|none|n/a)([ ,]|$)",
    "five_digit_token": r"(^|\D)\d{5}(\D|$)",
    "six_digit_token": r"(^|\D)\d{6}(\D|$)",
    "six_digit_spaced_pin": r"(^|\D)\d{3} \d{3}(\D|$)",
    "landmark_word": r"(?i)\b(near|opp|opposite|behind|beside|next to|nr)\b",
    "po_box": r"(?i)\bp\.?\s?o\.?\s?box\b",
    "unit_apt": r"(?i)\b(unit|apt|apartment|suite|ste|flat|floor)\b",
    "domain_like": r"(?i)\.(com|in|net|org|fr|co)\b",
    "legal_suffix": r"(?i)\b(inc|llc|ltd|limited|pvt|private|corp|corporation|co|company|llp|plc|sarl|sas|sa|sci|eurl|s\.a\.s|s\.a\.r\.l)\b\.?",
    "parenthesis": r"[()]",
    "ampersand": r"&",
}


def pattern_rates(arr: pa.Array, keys: list[str]) -> dict:
    return {k: rate(pc.match_substring_regex(arr, PATTERNS[k])) for k in keys}


NAME_PATTERN_KEYS = ["has_digit", "non_ascii", "devanagari", "other_indic_script", "latin_accented",
                     "leading_junk", "all_upper_latin", "literal_null_token", "domain_like",
                     "legal_suffix", "parenthesis", "ampersand"]
ADDR_PATTERN_KEYS = ["has_digit", "non_ascii", "devanagari", "other_indic_script", "latin_accented",
                     "all_upper_latin", "literal_null_token", "five_digit_token", "six_digit_token",
                     "six_digit_spaced_pin", "landmark_word", "po_box", "unit_apt"]


# --------------------------------------------------------------------------
# Per-source profile
# --------------------------------------------------------------------------
def profile_source(t: pa.Table) -> dict:
    n = t.num_rows
    out: dict = {"rows": n, "columns": t.column_names}
    eid = t["entity_id"]
    out["entity_id_unique"] = pc.count_distinct(eid).as_py()
    out["entity_id_prefixes"] = top_counts(pc.utf8_slice_codeunits(eid, 0, 3), 5)
    num_len = pc.utf8_length(pc.utf8_slice_codeunits(eid, 3, 100))
    out["entity_id_numeric_len"] = top_counts(num_len, 10)
    out["entity_id_non_numeric_suffix"] = rate(pc.invert(pc.match_substring_regex(eid, r"^S[123]-\d+$")))

    for col in ["business_name", "business_address", "country"]:
        a = t[col]
        trimmed = pc.utf8_trim_whitespace(a)
        out[f"{col}__empty_rate"] = rate(pc.equal(trimmed, ""))
        out[f"{col}__unique"] = pc.count_distinct(a).as_py()
        out[f"{col}__unique_normalized"] = pc.count_distinct(norm_text(a)).as_py() if col != "country" else None
    out["country_counts"] = top_counts(t["country"], 20)

    name, addr = t["business_name"], t["business_address"]
    out["name_char_len"] = qtiles(pc.utf8_length(name))
    out["name_token_count"] = qtiles(pc.list_value_length(tokens(name)))
    out["addr_char_len"] = qtiles(pc.utf8_length(addr))
    out["addr_comma_components"] = qtiles(pc.list_value_length(pc.split_pattern(addr, pattern=",")))
    out["name_patterns"] = pattern_rates(name, NAME_PATTERN_KEYS)
    out["addr_patterns"] = pattern_rates(addr, ADDR_PATTERN_KEYS)

    # Per-country breakdown of the most decision-relevant patterns
    per_c = {}
    for c, cnt in out["country_counts"]:
        sub = t.filter(pc.equal(t["country"], c))
        per_c[c] = {
            "rows": cnt,
            "name_empty": rate(pc.equal(pc.utf8_trim_whitespace(sub["business_name"]), "")),
            "addr_empty": rate(pc.equal(pc.utf8_trim_whitespace(sub["business_address"]), "")),
            "name": pattern_rates(sub["business_name"], ["non_ascii", "devanagari", "other_indic_script",
                                                         "leading_junk", "legal_suffix", "all_upper_latin"]),
            "addr": pattern_rates(sub["business_address"], ["devanagari", "other_indic_script", "five_digit_token",
                                                            "six_digit_token", "landmark_word",
                                                            "literal_null_token", "all_upper_latin"]),
            "addr_char_len_median": qtiles(pc.utf8_length(sub["business_address"]), (0.5,))["p50"],
        }
    out["per_country"] = per_c

    # Duplicates
    combo = pc.binary_join_element_wise(name, addr, t["country"], "\x1f")
    out["dup_exact_name_addr_country_rows"] = n - pc.count_distinct(combo).as_py()
    ncombo = pc.binary_join_element_wise(norm_text(name), norm_text(addr), t["country"], "\x1f")
    out["dup_normalized_name_addr_country_rows"] = n - pc.count_distinct(ncombo).as_py()
    nn = norm_text(name)
    out["top_normalized_names"] = top_counts(nn, 25)
    vc = pc.value_counts(nn).field("counts").to_numpy()
    out["normalized_name_freq"] = {
        "names_appearing_once_share_of_rows": round(float(vc[vc == 1].sum()) / n, 4),
        "rows_with_name_freq_ge_10": round(float(vc[vc >= 10].sum()) / n, 4),
        "rows_with_name_freq_ge_100": round(float(vc[vc >= 100].sum()) / n, 4),
    }
    out["examples_random"] = t.take(np.random.default_rng(0).choice(n, 12, replace=False)).to_pylist()
    return out


# --------------------------------------------------------------------------
# Ground truth
# --------------------------------------------------------------------------
def explode_gt(gt: pa.Table) -> pa.Table:
    lists = pc.split_pattern(gt["matched_entity_ids"], pattern=",")
    parent = pc.list_parent_indices(lists)
    flat = pc.utf8_trim_whitespace(pc.list_flatten(lists))
    s1 = pc.take(gt["source1_entity_id"], parent)
    pairs = pa.table({"s1": s1, "other": flat})
    return pairs.filter(pc.not_equal(pairs["other"], ""))


def profile_gt(gt: pa.Table, pairs: pa.Table) -> dict:
    out: dict = {"rows": gt.num_rows, "s1_unique": pc.count_distinct(gt["source1_entity_id"]).as_py()}
    other = pairs["other"]
    pref = pc.utf8_slice_codeunits(other, 0, 3)
    out["pair_count"] = pairs.num_rows
    out["pair_prefixes"] = top_counts(pref, 5)
    out["other_ids_unique"] = pc.count_distinct(other).as_py()
    out["other_ids_in_multiple_s1_lists"] = pairs.num_rows - out["other_ids_unique"]

    # per-S1 structure
    s1_all = gt["source1_entity_id"]
    is2 = pc.cast(pc.equal(pref, "S2-"), pa.int64())
    is3 = pc.cast(pc.equal(pref, "S3-"), pa.int64())
    agg = pa.table({"s1": pairs["s1"], "n2": is2, "n3": is3}).group_by("s1").aggregate(
        [("n2", "sum"), ("n3", "sum")])
    joined = pa.table({"s1": s1_all}).join(agg, "s1", join_type="left outer")
    n2 = pc.fill_null(joined["n2_sum"], 0).to_numpy()
    n3 = pc.fill_null(joined["n3_sum"], 0).to_numpy()
    tot = n2 + n3
    N = len(tot)
    out["singleton_rate"] = round(float((tot == 0).mean()), 6)
    out["matches_per_s1_distribution"] = {int(k): int(v) for k, v in zip(*np.unique(tot, return_counts=True))}
    out["matches_per_s1_stats"] = {"mean": round(float(tot.mean()), 3),
                                   "mean_nonsingleton": round(float(tot[tot > 0].mean()), 3),
                                   "median_nonsingleton": float(np.median(tot[tot > 0])), "max": int(tot.max())}
    out["one_to_one_rate"] = round(float((tot == 1).mean()), 6)
    out["one_to_many_rate"] = round(float((tot >= 2).mean()), 6)
    out["n_s2_per_s1_distribution"] = {int(k): int(v) for k, v in zip(*np.unique(n2, return_counts=True))}
    out["n_s3_per_s1_distribution"] = {int(k): int(v) for k, v in zip(*np.unique(n3, return_counts=True))}
    combos = {}
    key = np.minimum(n2, 4) * 10 + np.minimum(n3, 4)
    for k, v in zip(*np.unique(key, return_counts=True)):
        combos[f"S2={k // 10}{'+' if k // 10 == 4 else ''},S3={k % 10}{'+' if k % 10 == 4 else ''}"] = round(v / N, 5)
    out["s2_s3_combo_share"] = combos
    out["has_s2_match_rate"] = round(float((n2 > 0).mean()), 6)
    out["has_s3_match_rate"] = round(float((n3 > 0).mean()), 6)
    out["has_both_rate"] = round(float(((n2 > 0) & (n3 > 0)).mean()), 6)
    return out


# --------------------------------------------------------------------------
# Pair-level probes on a sample of ground-truth pairs
# --------------------------------------------------------------------------
def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def pair_probe_rows(p1: dict, p2: dict, tokfreq: dict, generic: set) -> dict:
    n1, n2 = p1["nn"], p2["nn"]
    t1, t2 = set(n1.split()), set(n2.split())
    t1s, t2s = t1 - generic, t2 - generic
    a1, a2 = set(p1["na"].split()), set(p2["na"].split())
    num1 = {x for x in a1 if any(ch.isdigit() for ch in x)}
    num2 = {x for x in a2 if any(ch.isdigit() for ch in x)}
    shared = t1s & t2s
    min_df = min((tokfreq.get(x, 0) for x in shared), default=None)
    ashared = (a1 & a2) - generic
    return {
        "country_equal": p1["country"] == p2["country"],
        "name_exact_raw": p1["business_name"] == p2["business_name"],
        "name_exact_norm": n1 == n2 and n1 != "",
        "name_sorted_tokens_equal": sorted(t1) == sorted(t2) and bool(t1),
        "name_jaccard": jaccard(t1, t2),
        "name_share_any_token": bool(t1 & t2),
        "name_share_nongeneric_token": bool(shared),
        "name_shared_token_min_df": min_df,
        "name_char3_share": bool({n1[i:i + 3] for i in range(len(n1) - 2)} & {n2[i:i + 3] for i in range(len(n2) - 2)}),
        "addr_exact_norm": p1["na"] == p2["na"] and p1["na"] != "",
        "addr_jaccard": jaccard(a1, a2),
        "addr_share_nongeneric_token": bool(ashared),
        "addr_share_numeric_token": bool(num1 & num2),
        "addr_either_empty": p1["na"] == "" or p2["na"] == "",
        "script_mismatch_name": (p1["name_nonascii"] != p2["name_nonascii"]),
        "name_either_empty": n1 == "" or n2 == "",
        "no_name_no_addr_overlap": not (t1 & t2) and not ((a1 & a2) - generic),
    }


def summarise_probes(rows: list[dict]) -> dict:
    out = {}
    keys = rows[0].keys()
    for k in keys:
        vals = [r[k] for r in rows if r[k] is not None]
        if not vals:
            continue
        if isinstance(vals[0], bool):
            out[k] = round(sum(vals) / len(rows), 4)
        elif k == "name_shared_token_min_df":
            arr = np.array(vals)
            out[k] = {f"p{q}": float(np.percentile(arr, q)) for q in (50, 75, 90, 95, 99)} | {
                "share_min_df_le_100_of_all_pairs": round(float((arr <= 100).sum()) / len(rows), 4),
                "share_min_df_le_1000_of_all_pairs": round(float((arr <= 1000).sum()) / len(rows), 4)}
        else:
            arr = np.array(vals, dtype=float)
            out[k] = {"mean": round(float(arr.mean()), 4), "p10": round(float(np.percentile(arr, 10)), 4),
                      "p50": round(float(np.percentile(arr, 50)), 4), "p90": round(float(np.percentile(arr, 90)), 4)}
    return out


def enrich(t: pa.Table) -> pa.Table:
    return t.append_column("nn", norm_text(t["business_name"])).append_column(
        "na", norm_text(t["business_address"])).append_column(
        "name_nonascii", pc.match_substring_regex(t["business_name"], r"[^\x00-\x7F]"))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="datasets")
    ap.add_argument("--out", default="reports/profiling/profile_summary.json")
    ap.add_argument("--sample-s1", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    R: dict = {"meta": {"seed": args.seed, "sample_s1": args.sample_s1, "pyarrow": pa.__version__}}
    rng = np.random.default_rng(args.seed)
    T0 = time.time()

    def save():
        out_path.write_text(json.dumps(R, indent=1, ensure_ascii=False, default=str), encoding="utf-8")

    def log(msg):
        print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)

    # ---- ground truth first (drives sampling) --------------------------
    gt = read_tsv(d / "train" / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"])
    R["file_line_counts"] = {}
    pairs = explode_gt(gt)
    R["ground_truth"] = profile_gt(gt, pairs)
    log("ground truth profiled")
    s1_ids_gt = set(gt["source1_entity_id"].to_pylist())

    # sample S1 entities for pair probes (from ALL S1, singletons included)
    samp_idx = rng.choice(gt.num_rows, size=min(args.sample_s1, gt.num_rows), replace=False)
    samp_s1 = gt["source1_entity_id"].take(pa.array(samp_idx))
    samp_pairs = pairs.filter(pc.is_in(pairs["s1"], value_set=samp_s1))
    all_other_ids = pairs["other"]
    del gt
    gc.collect()

    records: dict[str, dict] = {}
    row_index: dict[str, int] = {}
    overlap: dict = {}
    s1_nonsingleton = set(samp_pairs["s1"].to_pylist())

    for split in ["train", "test"]:
        for s in (1, 2, 3):
            path = d / split / f"{split}_source{s}.tsv"
            t = read_tsv(path, SOURCE_COLS)
            key = f"{split}_source{s}"
            R["file_line_counts"][key] = {"lines_incl_header": count_lines(path), "parsed_rows": t.num_rows,
                                          "size_bytes": path.stat().st_size}
            R[key] = profile_source(t)
            log(f"{key} profiled ({t.num_rows} rows)")

            ids = t["entity_id"]
            if split == "train":
                if s == 1:
                    fs = set(ids.to_pylist())
                    R["ground_truth"]["s1_file_vs_gt"] = {"in_file_not_gt": len(fs - s1_ids_gt),
                                                          "in_gt_not_file": len(s1_ids_gt - fs)}
                    del fs
                else:
                    pref = f"S{s}-"
                    gt_ids_s = all_other_ids.filter(pc.starts_with(all_other_ids, pref))
                    present = pc.is_in(gt_ids_s, value_set=ids)
                    matched_mask = pc.is_in(ids, value_set=gt_ids_s)
                    R[key]["gt_ids_missing_from_file"] = int(len(gt_ids_s) - pc.sum(pc.cast(present, pa.int64())).as_py())
                    R[key]["share_of_records_matched_to_some_s1"] = rate(matched_mask)
                    # Are unmatched records different in character? (distractor analysis)
                    unm = t.filter(pc.invert(matched_mask))
                    R[key]["unmatched_records_country_counts"] = top_counts(unm["country"], 10)
                    R[key]["unmatched_examples"] = unm.take(np.random.default_rng(1).choice(unm.num_rows, 8, replace=False)).to_pylist()
                    del unm
                # keep sampled records + row positions (row-order leakage probe)
                want = samp_s1 if s == 1 else samp_pairs["other"]
                mask = pc.is_in(ids, value_set=want)
                idx = np.nonzero(mask.to_numpy(zero_copy_only=False))[0]
                sub = enrich(t.take(pa.array(idx)))
                for i, r in zip(idx, sub.to_pylist()):
                    records[r["entity_id"]] = r
                    row_index[r["entity_id"]] = int(i)
            # train/test overlap bookkeeping (ids + exact normalised record keys)
            combo = pc.binary_join_element_wise(norm_text(t["business_name"]), norm_text(t["business_address"]),
                                                t["country"], "\x1f")
            overlap[key] = {"ids": hash_arr(ids), "combo": hash_arr(pc.unique(combo))}
            del t
            gc.collect()

        # ---- after train sources loaded: token doc-freq on candidate side ----
        if split == "train":
            log("computing sampled pair probes")
            # token frequency over S2+S3 names (restricted to tokens that appear in sample)
            samp_tokens = set()
            for r in records.values():
                samp_tokens.update(r["nn"].split())
            tokfreq: dict = {}
            for s in (2, 3):
                t = read_tsv(d / "train" / f"train_source{s}.tsv", ["entity_id", "business_name", "business_address", "country"])
                toks = pc.list_flatten(pc.list_unique(tokens(t["business_name"])) if hasattr(pc, "list_unique") else tokens(t["business_name"]))
                toks = toks.filter(pc.is_in(toks, value_set=pa.array(list(samp_tokens))))
                vc = pc.value_counts(toks)
                for v, c in zip(vc.field("values").to_pylist(), vc.field("counts").to_pylist()):
                    tokfreq[v] = tokfreq.get(v, 0) + c
                del t, toks
                gc.collect()
            generic = set(GENERIC_TOKENS)
            R["top_s2s3_name_token_freq_in_sample_vocab"] = sorted(tokfreq.items(), key=lambda x: -x[1])[:40]

            pos_rows, by_country, by_src = [], {}, {"S2-": [], "S3-": []}
            examples = []
            for s1, o in zip(samp_pairs["s1"].to_pylist(), samp_pairs["other"].to_pylist()):
                if s1 not in records or o not in records:
                    continue
                pr = pair_probe_rows(records[s1], records[o], tokfreq, generic)
                pos_rows.append(pr)
                by_country.setdefault(records[s1]["country"], []).append(pr)
                by_src[o[:3]].append(pr)
                if len(examples) < 40:
                    examples.append({"s1": {k: records[s1][k] for k in SOURCE_COLS},
                                     "match": {k: records[o][k] for k in SOURCE_COLS}})
            # negative contrast: random S1 (non-singleton sample) x random other record, same country
            others = [r for k, r in records.items() if not k.startswith("S1-")]
            s1recs = [r for k, r in records.items() if k.startswith("S1-")]
            others_by_c: dict = {}
            for r in others:
                others_by_c.setdefault(r["country"], []).append(r)
            neg_rows = []
            nrng = np.random.default_rng(args.seed + 1)
            for _ in range(min(50_000, len(pos_rows))):
                a = s1recs[nrng.integers(len(s1recs))]
                pool = others_by_c.get(a["country"])
                if not pool:
                    continue
                b = pool[nrng.integers(len(pool))]
                neg_rows.append(pair_probe_rows(a, b, tokfreq, generic))
            R["pair_probes"] = {
                "n_positive_pairs": len(pos_rows),
                "positive": summarise_probes(pos_rows),
                "positive_by_s1_country": {c: summarise_probes(v) | {"n": len(v)} for c, v in by_country.items()},
                "positive_by_target_source": {c: summarise_probes(v) | {"n": len(v)} for c, v in by_src.items() if v},
                "random_same_country_negative": summarise_probes(neg_rows),
                "examples": examples,
            }
            # singletons in sample: what do they look like?
            singles = [records[x] for x in samp_s1.to_pylist() if x not in s1_nonsingleton and x in records]
            R["singleton_examples"] = [{k: r[k] for k in SOURCE_COLS} for r in singles[:25]]
            sc: dict = {}
            for r in singles:
                sc[r["country"]] = sc.get(r["country"], 0) + 1
            s1c: dict = {}
            for r in s1recs:
                s1c[r["country"]] = s1c.get(r["country"], 0) + 1
            R["singleton_rate_by_country_sample"] = {c: round(sc.get(c, 0) / s1c[c], 4) for c in s1c}

            # row-order / ID leakage probe
            s1_pos = np.array([row_index[s] for s, o in zip(samp_pairs["s1"].to_pylist(), samp_pairs["other"].to_pylist()) if s in row_index and o in row_index])
            o_pos = np.array([row_index[o] for s, o in zip(samp_pairs["s1"].to_pylist(), samp_pairs["other"].to_pylist()) if s in row_index and o in row_index])
            s1_num = np.array([int(s[3:]) for s, o in zip(samp_pairs["s1"].to_pylist(), samp_pairs["other"].to_pylist()) if s in row_index and o in row_index])
            o_num = np.array([int(o[3:]) for s, o in zip(samp_pairs["s1"].to_pylist(), samp_pairs["other"].to_pylist()) if s in row_index and o in row_index])

            def spearman(x, y):
                rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
                return round(float(np.corrcoef(rx, ry)[0, 1]), 5)
            R["leakage_probes"] = {
                "spearman_rowpos_s1_vs_match": spearman(s1_pos, o_pos),
                "spearman_idnum_s1_vs_match": spearman(s1_num, o_num),
                "share_same_id_number": round(float((s1_num == o_num).mean()), 6),
            }
            del records, row_index
            gc.collect()
            log("pair probes done")

    # ---- train/test overlap ------------------------------------------------
    ov = {}
    for s in (1, 2, 3):
        a, b = overlap[f"train_source{s}"], overlap[f"test_source{s}"]
        ov[f"source{s}"] = {
            "id_overlap": int(np.isin(b["ids"], a["ids"]).sum()),
            "normalized_record_overlap_unique": int(np.isin(b["combo"], a["combo"]).sum()),
            "test_unique_normalized_records": len(b["combo"]),
        }
    # cross-source id collisions (numeric part) within test
    R["train_test_overlap"] = ov
    R["meta"]["elapsed_s"] = round(time.time() - T0, 1)
    save()
    log(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
