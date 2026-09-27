"""EXP-B-01 hypothesis probe (read-only): identical name_core / sorted tokens / leading-zero numbers
among found vs non-candidate true pairs, and legal-form / house-number conflicts among
predicted TP vs FP pairs. Writes experiments/runs/EXP-B-01/analysis/hypothesis_probe.json.

    python scripts/analysis/hypothesis_probe.py
"""
import sys, re, json, numpy as np, pyarrow as pa, pyarrow.compute as pc, pyarrow.parquet as pq
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from analyze_exp_b01 import load_views, code
from ber.config import load_config, Paths
from ber.data.split import load_val_ids
from ber.data.normalize_records import load_manifest
from ber.pipeline.run_baseline import _ids, pool_ids
from ber.normalization.views import LEGAL_FORMS
from ber.utils.runtime import configure_arrow
configure_arrow()
cfg = load_config("configs/baseline_v0.yaml"); paths = Paths(cfg)
s1_ids = _ids(paths, "train", "source1"); p_ids = pool_ids(paths, "train")
n_s2 = int(load_manifest(paths, "train")["source2"]["rows"])
val_arr = pa.array(load_val_ids(paths), pa.string())
gt = pq.read_table(paths.interim_file("train", "ground_truth"))
gt = gt.filter(pc.is_in(gt.column("source1_entity_id"), value_set=val_arr))
L = pc.split_pattern(gt.column("matched_entity_ids"), ","); fl = pc.list_flatten(L); par = pc.list_parent_indices(L)
k = pc.not_equal(fl, ""); fl = pc.filter(fl, k); par = pc.filter(par, k)
gs = pc.index_in(pc.take(gt.column("source1_entity_id"), par), value_set=s1_ids).to_numpy(zero_copy_only=False).astype(np.int64)
gp = pc.index_in(fl, value_set=p_ids).to_numpy(zero_copy_only=False).astype(np.int64)
t = pa.concat_tables([pq.read_table(p, columns=["s1_row","pool_row","score","name_norm_eq","name_core_eq"]) for p in sorted(paths.scored_dir("train").glob("*.parquet"))])
cc = code(t.column("s1_row").to_numpy(), t.column("pool_row").to_numpy())
found = np.isin(code(gs, gp), cc)
v1 = load_views(paths, "train", "source1", np.unique(gs))
v2 = load_views(paths, "train", "source2", gp[gp < n_s2]); v3 = load_views(paths, "train", "source3", gp[gp >= n_s2] - n_s2)
def pv(p): return v2.get(p) if p < n_s2 else v3.get(p - n_s2)
strip0 = lambda s: {x.lstrip("0") or "0" for x in s.split()}
stats = {k: [0, 0] for k in ["core_eq","sorted_core_eq","num_overlap_raw","num_overlap_strip0","pool_has_saint","s1_has_street","sorted_norm_eq"]}
n = [0, 0]
for s, p, f in zip(gs, gp, found):
    a, b = v1[int(s)], pv(int(p)); i = 0 if f else 1; n[i] += 1
    stats["core_eq"][i] += a[1] == b[1]
    stats["sorted_core_eq"][i] += sorted(a[1].split()) == sorted(b[1].split())
    stats["sorted_norm_eq"][i] += sorted(a[0].split()) == sorted(b[0].split())
    stats["num_overlap_raw"][i] += bool(set(a[4].split()) & set(b[4].split()))
    stats["num_overlap_strip0"][i] += bool(strip0(a[4]) & strip0(b[4])) if a[4] and b[4] else False
    stats["pool_has_saint"][i] += "saint" in b[3].split()
    stats["s1_has_street"][i] += "street" in a[3].split()
res = {k: {"found": v[0]/n[0], "not_in_candidates": v[1]/n[1]} for k, v in stats.items()}
res["n"] = {"found": n[0], "not_in_candidates": n[1]}
# legal-form conflict among predicted pairs (TP vs FP), threshold 0.67
t2 = pa.concat_tables([pq.read_table(p, columns=["s1_row","pool_row","score"]) for p in sorted(paths.scored_dir("train").glob("*.parquet"))])
sr, pr, scv = t2.column("s1_row").to_numpy(), t2.column("pool_row").to_numpy().astype(np.int64), t2.column("score").to_numpy()
tru = np.isin(code(sr, pr), code(gs, gp)); pred = scv >= 0.67
need = np.unique(pr[pred]); w2 = load_views(paths, "train", "source2", need[need < n_s2]); w3 = load_views(paths, "train", "source3", need[need >= n_s2] - n_s2)
v1b = load_views(paths, "train", "source1", np.unique(sr[pred]))
def lf(norm): return {t for t in norm.split() if t in LEGAL_FORMS} - {"co","company","private"}
canon = {"ltd":"ltd","limited":"ltd","inc":"inc","incorporated":"inc","corp":"corp","corporation":"corp"}
def conflict(a, b):
    x = {canon.get(t, t) for t in lf(a)}; y = {canon.get(t, t) for t in lf(b)}
    return bool(x) and bool(y) and not (x & y)
def numconf(a, b):
    x, y = strip0(a), strip0(b)
    return bool(a) and bool(b) and not (x & y)
cnt = {"TP": [0,0,0], "FP": [0,0,0]}
for i in np.flatnonzero(pred):
    a = v1b[int(sr[i])]; p = int(pr[i]); b = w2.get(p) if p < n_s2 else w3.get(p - n_s2)
    kk = "TP" if tru[i] else "FP"; cnt[kk][0] += 1
    cnt[kk][1] += conflict(a[0], b[0]); cnt[kk][2] += numconf(a[4], b[4])
res["predicted_pairs_legal_form_conflict"] = {k: v[1]/v[0] for k, v in cnt.items()}
res["predicted_pairs_no_shared_number_both_have_numbers"] = {k: v[2]/v[0] for k, v in cnt.items()}
res["predicted_counts"] = {k: v[0] for k, v in cnt.items()}
json.dump(res, open("experiments/runs/EXP-B-01/analysis/hypothesis_probe.json", "w"), indent=2)
print(json.dumps(res, indent=1))
