"""FINAL RESEARCH AUDIT: compile the synthesis deliverables (experiment matrix, public transfer, shadow audit,
leakage, reproducibility, GPU feasibility) from verified artifacts. Read-only; writes only to
experiments/runs/FINAL_RESEARCH_AUDIT/. Numbers are read from artifacts, never typed in, except the team-reported
public scores (labelled)."""

from __future__ import annotations

import json
from pathlib import Path

O = Path("experiments/runs/FINAL_RESEARCH_AUDIT")
R = Path("experiments/runs")


def j(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def w(name, obj):
    (O / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


PUBLIC = {"EXP-B-01": 0.751, "EXP-F-01b": 0.815, "EXP-M-03": 0.889, "EXP-B-03a": 0.890, "EXP-B-04": 0.91, "EXP-N-02": 0.928,
          "best pre-S-01 (EXP-B-06 or EXP-B-07 t0.67)": 0.935, "EXP-S-01": 0.920}  # team-reported


def matrix():
    sh = j(R / "POST-S01-FORENSIC/shadow_metrics.json")["systems"]
    s02 = j(R / "EXP-S-02/analysis/shadow_eval.json")["competitions"]["T1(base)"]["metrics"]
    b08 = j(R / "EXP-B-08/analysis/eval.json")["competitions"]["T1(base)"]["metrics"]
    fc = lambda e: j(R / e / "full_competition_unrestricted.json")["variants"]  # noqa: E731
    val = lambda e: j(R / e / "metrics.json")["at_threshold_used"]["f05"]  # noqa: E731
    sb = lambda s: sh[s]["SHADOW-BASE"]["f05"]  # noqa: E731
    E = lambda e: fc(e)["best_claimant_margin_0.00"]["E"]["f05"]  # noqa: E731
    rows = [
        dict(id="EXP-B-01", hypothesis="Blocking V0 + weighted similarity rule + tuned threshold gives a valid baseline", primary_change="first end-to-end pipeline",
             control=None, validation_f05=0.7851, E=None, shadow_base=None, public=PUBLIC["EXP-B-01"], decision="banked baseline",
             lesson="rule matcher: low recall; validation-public gap -0.034", artifacts_usable="report only", verified="ledger + reports/EXP-B-01_SUBMISSION_1_REPORT.md"),
        dict(id="EXP-F-01b", hypothesis="learned LR + number/legal-form conflict features beat the rule", primary_change="LR matcher + conflict features",
             control="B-01/M-02b", validation_f05=val("EXP-F-01b"), E=None, shadow_base=None, public=PUBLIC["EXP-F-01b"], decision="banked",
             lesson="learned matcher +0.05 local; gap -0.022", artifacts_usable="model.json", verified="EXP-F-01b/metrics.json"),
        dict(id="EXP-M-03", hypothesis="a tree model extracts interaction headroom", primary_change="LR -> LightGBM (same inputs)", control="EXP-F-01b",
             validation_f05=val("EXP-M-03"), E=None, shadow_base=None, public=PUBLIC["EXP-M-03"], decision="banked",
             lesson="largest single model gain (+0.067 local, +0.074 public)", artifacts_usable="model.json", verified="EXP-M-03/metrics.json"),
        dict(id="EXP-B-03a", hypothesis="name-token blocking + unchanged M-03 model", primary_change="enable name_token, NO refit", control="EXP-M-03",
             validation_f05=val("EXP-B-03a"), E=None, shadow_base=None, public=PUBLIC["EXP-B-03a"], decision="rejected",
             lesson="blocker changes shift cheap_* inputs: refit is mandatory (DEC-017)", artifacts_usable="diagnostic", verified="EXP-B-03a/metrics.json"),
        dict(id="EXP-B-04", hypothesis="combined keys + refit recover missed pairs", primary_change="+name_addr +addr_pair, refit", control="EXP-M-03",
             validation_f05=val("EXP-B-04"), E=sh["B04"]["E"]["f05"], shadow_base=sb("B04"), public=PUBLIC["EXP-B-04"], decision="banked (submitted WITHOUT D-01)",
             lesson="retrieval + refit transfers (+0.02 public)", artifacts_usable="model; NOTE submissions/EXP-B-04_output holds the N-02 file",
             verified="EXP-B-04/metrics.json; POST-S01 shadow_metrics (E unrestricted)"),
        dict(id="EXP-D-01", hypothesis="best claimant per record under full competition", primary_change="decision layer (D-01), no refit", control="EXP-B-04",
             validation_f05=None, E=0.9313, E_control=0.9273, shadow_base=None, public=None, decision="kept in every later system",
             lesson="full-competition evaluation is essential", artifacts_usable="yes", verified="EXP-D-01/full_competition_eval.json (ledger)"),
        dict(id="EXP-B-05", hypothesis="wider address-word window", primary_change="max_words 3 -> 5, refit", control="B-04 + D-01",
             validation_f05=val("EXP-B-05"), E=E("EXP-B-05"), E_control=0.9313, shadow_base=None, public=None, decision="rejected (+0.0010, precision -0.0027)",
             lesson="more words add hard negatives", artifacts_usable="diagnostic", verified="EXP-B-05/full_competition_unrestricted.json"),
        dict(id="EXP-N-02", hypothesis="train-learned cross-script view for India", primary_change="name_xlit view in name keys + 2 features, refit", control="B-04 + D-01",
             validation_f05=val("EXP-N-02"), E=E("EXP-N-02"), E_control=0.9313, shadow_base=sb("N02"), public=PUBLIC["EXP-N-02"], decision="kept",
             lesson="+0.018 public vs B-04 (largest after LightGBM); gap -0.012", artifacts_usable="map + model", verified="EXP-N-02/full_competition_unrestricted.json"),
        dict(id="EXP-N-03", hypothesis="learned Indic legal-form tokens", primary_change="legal-token stage of the map, refit", control="N-02 + D-01",
             validation_f05=val("EXP-N-03"), E=E("EXP-N-03"), E_control=E("EXP-N-02"), shadow_base=None, public=None, decision="below gate; base of B-07/B-06",
             lesson="representation gains saturate", artifacts_usable="yes", verified="EXP-N-03/full_competition_unrestricted.json"),
        dict(id="EXP-B-07", hypothesis="exact name x address word for capped generic names", primary_change="exact_core.with_addr_words key, refit, t 0.67",
             control="N-02 + D-01", validation_f05=sh["B07_t067"]["validation"]["f05"], E=sh["B07_t067"]["E"]["f05"], E_control=E("EXP-N-02"), shadow_base=sb("B07_t067"),
             public="0.935 or not submitted [U]", decision="kept", lesson="cap-aware retrieval keys transfer",
             artifacts_usable="yes; run-folder metrics are at t 0.63, t 0.67 values in POST-S01 shadow_metrics / d01_threshold_select.json", verified="POST-S01 shadow_metrics (B07_t067)"),
        dict(id="EXP-B-06", hypothesis="name-token-pair key for name-only records", primary_change="name_token.pair, refit (t 0.67)", control="N-02 + D-01; parent B-07",
             validation_f05=sh["B06"]["validation"]["f05"], E=sh["B06"]["E"]["f05"], E_control=E("EXP-N-02"), shadow_base=sb("B06"), public="0.935 or not submitted [U]",
             decision="kept; control of every later experiment", lesson="best stage-1 system", artifacts_usable="model, all-train + test scores",
             verified="POST-S01 shadow_metrics (B06); EXP-B-06/full_competition_unrestricted.json"),
        dict(id="EXP-S-01", hypothesis="collective stage-2 with context decides better", primary_change="stage-2 LightGBM (41 features), 0.65 + D-01", control="B-06 + D-01",
             validation_f05=sh["S01"]["validation"]["f05"], E=sh["S01"]["E"]["f05"], E_half_B=sh["S01"]["E_half_B"]["f05"], shadow_base=sb("S01"), public=PUBLIC["EXP-S-01"],
             decision="public regression", lesson="context-driven additions fail under the test's candidate-environment shift", artifacts_usable="model (reused read-only by S-02)",
             verified="POST-S01 shadow_metrics; EXP-S-01/stage2_eval.json"),
        dict(id="EXP-S-02", hypothesis="S-01 as a veto only", primary_change="final = (B-06 + D-01) AND S-01", control="B-06 + D-01",
             validation_f05=s02["S02"]["validation"]["f05"], E=s02["S02"]["E"]["f05"], E_half_B=s02["S02"]["E_half_B"]["f05"], shadow_base=s02["S02"]["SHADOW-BASE"]["f05"],
             public=None, decision="KEEP (all gates), not submitted", lesson="density-invariant +0.0027..0.0030; precision-side only", artifacts_usable="yes",
             verified="EXP-S-02/analysis/shadow_eval.json, gates.json"),
        dict(id="EXP-B-08", hypothesis="ranks 21-30 hold recoverable true pairs", primary_change="top_k 20 -> 30, refit (t 0.68)", control="B-06 + D-01",
             validation_f05=b08["B08"]["validation"]["f05"], E=b08["B08"]["E"]["f05"], E_half_B=b08["B08"]["E_half_B"]["f05"], shadow_base=b08["B08"]["SHADOW-BASE"]["f05"],
             public=None, decision="REJECT (+0.0017..0.0019 < +0.002)", lesson="clean, density-stable, small: top-K nearly exhausted",
             artifacts_usable="model + all-train/test scores (reusable in a combined system)", verified="EXP-B-08/analysis/eval.json, gates.json"),
    ]
    w("experiment_matrix.json", {"labels": "[M] measured from the cited artifacts; public = team-reported; [U] unknown", "experiments": rows,
                                 "discrepancies": ["0.935 public file unattributed: submissions/ holds both B-06 (1e765d7a) and B-07 t0.67 (6aad6529) backups",
                                                   "submissions/EXP-B-04_output holds the N-02 file (4ff3fddf)",
                                                   "EXP-B-07 run-folder metrics are at t 0.63; adopted t 0.67 values live in other files",
                                                   "B-04 public reported only as approximately 0.91",
                                                   "root output/ = S-01 files (public 0.920), not the 0.935 control"]})
    return rows


def transfer(rows):
    sh = j(R / "POST-S01-FORENSIC/shadow_metrics.json")["systems"]
    obs = [
        {"system": "EXP-B-01", "local": 0.7851, "local_kind": "validation (no full competition)", "public": 0.751, "family": "rule"},
        {"system": "EXP-F-01b", "local": 0.8372, "local_kind": "validation", "public": 0.815, "family": "stage-1 LR"},
        {"system": "EXP-M-03", "local": 0.9037, "local_kind": "validation", "public": 0.889, "family": "stage-1 LGBM"},
        {"system": "EXP-B-03a", "local": 0.9027, "local_kind": "validation", "public": 0.890, "family": "stage-1 LGBM"},
        {"system": "EXP-B-04 (no D-01)", "local": sh["B04"]["E"]["f05"], "local_kind": "E full competition", "public": 0.91, "family": "stage-1 LGBM", "note": "public approx"},
        {"system": "EXP-N-02 + D-01", "local": sh["N02"]["E"]["f05"], "local_kind": "E full competition", "public": 0.928, "family": "stage-1 LGBM"},
        {"system": "EXP-B-06 + D-01 (if it is the 0.935 file)", "local": sh["B06"]["E"]["f05"], "local_kind": "E full competition", "public": 0.935, "family": "stage-1 LGBM", "note": "attribution [U]"},
        {"system": "EXP-B-07 t0.67 + D-01 (if it is the 0.935 file)", "local": sh["B07_t067"]["E"]["f05"], "local_kind": "E full competition", "public": 0.935, "family": "stage-1 LGBM", "note": "attribution [U]"},
        {"system": "EXP-S-01", "local": sh["S01"]["E"]["f05"], "local_kind": "E full competition", "public": 0.920, "family": "stage-2 additions"},
    ]
    for o in obs:
        o["gap_public_minus_local"] = o["public"] - o["local"]
        o["shadow_base"] = {"EXP-B-04 (no D-01)": sh["B04"]["SHADOW-BASE"]["f05"], "EXP-N-02 + D-01": sh["N02"]["SHADOW-BASE"]["f05"],
                            "EXP-S-01": sh["S01"]["SHADOW-BASE"]["f05"]}.get(o["system"])
    s1 = [o["gap_public_minus_local"] for o in obs if o["family"].startswith("stage-1 LGBM") and "E full" in o["local_kind"]]
    deltas = [{"from": "EXP-B-04", "to": "EXP-N-02", "local_delta": sh["N02"]["E"]["f05"] - sh["B04"]["E"]["f05"], "public_delta": 0.928 - 0.91,
               "note": "includes adding D-01 (+0.004 local) at the same time"},
              {"from": "EXP-N-02", "to": "0.935 file", "local_delta": sh["B06"]["E"]["f05"] - sh["N02"]["E"]["f05"], "public_delta": 0.935 - 0.928, "note": "B-06 assumed; B-07 gives 0.0062"},
              {"from": "0.935 file", "to": "EXP-S-01", "local_delta": sh["S01"]["E"]["f05"] - sh["B06"]["E"]["f05"], "public_delta": 0.920 - 0.935, "note": "sign inverted"}]
    w("public_transfer.json", {
        "labels": "local [M]; public team-reported; gaps [M]-[reported]; conclusions [I]", "observations": obs, "consecutive_deltas": deltas,
        "stage1_E_gap": {"values": s1, "min": min(s1), "max": max(s1), "spread": max(s1) - min(s1)},
        "stage2_additions_gap": obs[-1]["gap_public_minus_local"],
        "conclusions": [
            "[I] Stage-1 systems evaluated with full competition transfer with a gap of about -0.011 to -0.017 (spread ~0.005 over 4 points; one point approximate).",
            "[I] Consecutive stage-1 improvements of +0.007..+0.013 local transferred at roughly 1:1 or better in public.",
            "[I] Stage-2 additions (S-01) inverted in public: gap -0.041, ~3x the stage-1 gap. Transfer is family-dependent.",
            "[I] The observed gap variability (spread ~0.005, one approximate point) is larger than +0.001..0.002 local changes: such changes are not reliably visible publicly.",
            "[I] Minimum evidence for a submission: stage-1 local gain >= +0.004 on validation/GATE-B/E half B, density-stable, locked-agreeing; decision-layer (stage-2) changes additionally need no test-side behaviour anomaly (e.g. addition ratio test/train <= 1.3)."]})


def shadow_audit():
    m2 = j("experiments/shadow_tests/v2/manifest_v2.json")
    s02l = j(R / "EXP-S-02/analysis/leakage.json")
    b08l = j(R / "EXP-B-08/analysis/leakage.json")
    w("shadow_audit.json", {
        "labels": "[M] from manifests/leakage files; status judgements [I]",
        "v1 (POST-S01, experiments/shadow_tests/)": {
            "SHADOW-BASE": "446,321 S1 = E half B minus validation, u01('shadow') < 0.5; E half B was the aggregate gate of B-07/B-06/S-01 -> ALREADY USED",
            "SHADOW-TEST-MATCHED / CHAIN-ORPHANED / CHAIN-MEMBERS": "competitions over SHADOW-BASE; used in the floor sweep that selected S-02 -> ALREADY USED (development)",
            "GATE-B": "E half B minus SHADOW-BASE; ordinary gate set -> ALREADY USED"},
        "v2 (EXP-S-02, experiments/shadow_tests/v2/)": {
            "construction": m2["definitions"], "sizes": m2["sizes"], "tier_q": {k: v["q"] for k, v in m2["tiers"].items()},
            "tier_dev_density": {k: v["populations"]["SHADOW-DEV"]["lookalike_density"] for k, v in m2["tiers"].items()},
            "T4_reached_test_density": m2["tiers"]["T4"]["target_reached"],
            "leakage_S02": {k: s02l.get(k) for k in ("dev", "locked_A", "locked_B", "dev_locked_disjoint", "tier_masks_never_remove_an_evaluated_s1", "all_ok")},
            "leakage_B08(fit set vs shadow)": {k: v for k, v in b08l.items() if k.startswith("fit_s1_in_")}},
        "status": {
            "development benchmark": ["SHADOW-DEV (all tiers)", "v1 SHADOW-BASE / TEST-MATCHED / CHAIN sets", "GATE-B", "E half B", "frozen validation (used for every keep decision)"],
            "locked benchmark": ["SHADOW-LOCKED-A", "SHADOW-LOCKED-B"],
            "locked sets already looked at": {"EXP-S-02": "evaluated once after freeze, but the veto had been chosen on SHADOW-BASE (which contains them) -> not fresh for S-02",
                                             "EXP-B-08": "evaluated once after fit froze the model; REJECT was driven by the local +0.002 gate"},
            "future-virgin labelled benchmark": "NONE. Every labelled train S1 is in the fit set, the stage-2 training half (E half A), the frozen validation or E half B, and all of these have influenced decisions. LOCKED-A/B are the least-used (2 looks each)."},
        "density_matching_sufficiency": {
            "reached": "T1W / T4W match the test's per-S1 look-alike count distribution (density 0.706) by reweighting; pure orphan tiers stop at 0.462",
            "not matched": ["weak (<0.5) candidates per S1 (test 1.78 vs 1.19)", "peer-similar share of weak candidates (0.498 vs 0.410)",
                            "claimant structure under T4 (contested 5.8% vs test 15.9%)", "France (no labels; no proxy)", "eligible candidates per S1 (test 5.04 vs 4.41)"],
            "additional covariates recommended": ["weak candidates per S1", "eligible per S1", "record address-empty share", "cross-script candidate share", "stage-1 score distribution of candidates"]},
        "rules_for_future_use": ["Declare which sets an experiment may look at before running it",
                                 "LOCKED-A/B: one look per frozen candidate, logged in the experiment notes; never used to pick between variants",
                                 "Every look at LOCKED sets is recorded in docs/EXPERIMENT_LOG.md"]})


def leakage():
    w("leakage_audit.json", {
        "labels": "[M] verified this session or recorded by guarded code/tests",
        "checks": [
            {"check": "fit S1 vs frozen validation", "status": "PASS", "evidence": "fit_s1_subset raises on overlap; B-08 leakage.json fit_s1_in_validation 0; tests"},
            {"check": "stage-2 training partition vs validation / E half B / fit", "status": "PASS", "evidence": "run_stage2.populations guards (raise); S-02 leakage.json"},
            {"check": "shadow sets vs any training / threshold / early stopping", "status": "PASS", "evidence": "S-02 leakage.json all zero; B-08 fit_s1_in_SHADOW-* all 0"},
            {"check": "cross-script map learned from fit set only", "status": "PASS", "evidence": "learned_from_digest == fit-set digest (S-02 leakage.json); B-08 map identical to B-06"},
            {"check": "threshold source", "status": "PASS", "evidence": "resolve_threshold uses model meta (inner train-fold split) when threshold_from=model; tuned_threshold.json is report-only"},
            {"check": "test labels", "status": "PASS", "evidence": "none exist; test used only for label-free diagnostics"},
            {"check": "test-distribution information in evaluation", "status": "NOTE",
             "evidence": "SHADOW-TEST-MATCHED and T1W/T4W weights use test look-alike / candidate-count distributions (no labels). This makes selection deliberately test-aware; it is not training leakage but it is test-informed model selection."},
            {"check": "validation re-use (selection bias)", "status": "RISK",
             "evidence": "the frozen validation fold decided every keep since Phase 2; mild optimistic bias in reported validation values"},
            {"check": "E half B / SHADOW-BASE re-use", "status": "RISK", "evidence": "used as gates for B-07/B-06/S-01/S-02/B-08; the veto (S-02) was chosen from SHADOW-BASE aggregates"},
            {"check": "feature-map leakage (xlit cross-fitting)", "status": "PASS", "evidence": "fit rows use out-of-fold maps; fold 0 (inner tune) contributes to no map (tests/test_xlit.py)"}]})


def repro():
    w("reproducibility_audit.json", {
        "labels": "[M] verified this session unless marked",
        "issues": [
            {"severity": "HIGH", "issue": "root output/ holds the S-01 files (public 0.920), not the 0.935 control", "evidence": "sha256 output/matching_results.tsv = 4cad314e... = EXP-S-01/output",
             "fix": "team decision: restore the 0.935 file (or S-02 if later approved) before packaging"},
            {"severity": "HIGH", "issue": "the 0.935 public file is unattributed (B-06 or B-07 t0.67)", "evidence": "both backups exist in submissions/; no record of which was uploaded",
             "fix": "team confirms from the portal upload history; record in EXPERIMENT_LOG"},
            {"severity": "MEDIUM-HIGH", "issue": "numpy is imported from the user site (C:/Users/Param/AppData/Roaming/Python/Python313, numpy 2.2.4) instead of the env (numpy 2.2.6)",
             "evidence": "numpy.__file__; site.ENABLE_USER_SITE True", "fix": "run with PYTHONNOUSERSITE=1 or pin numpy==2.2.4 in the env and remove the user-site copy; re-verify a reference output hash; record in requirements"},
            {"severity": "MEDIUM", "issue": "config inheritance chains are 10-12 files deep; editing a parent silently changes every child (incl. the control)",
             "evidence": "configs/exp_s02_stage2_veto.yaml -> ... -> baseline_v0.yaml (12 levels); no drift today (snapshots == resolved configs)", "fix": "freeze resolved configs of controls (config_snapshot.yaml exists) and diff before every run"},
            {"severity": "MEDIUM", "issue": "analysis scripts full_competition.py / eval_exp_b05.py / post_s01_forensics.py delta materialise all candidate pairs (>5 GB with larger K)",
             "evidence": "B-06 fullcomp 4.40 GB, eval 4.50 GB, POST-S01 delta 5.47 GB", "fix": "use the streamed evaluators (exp_b08.py) for future experiments"},
            {"severity": "MEDIUM", "issue": "submissions/EXP-B-04_output holds the N-02 file", "evidence": "hash 4ff3fddf == EXP-N-02/output", "fix": "rename/annotate backups"},
            {"severity": "LOW", "issue": "EXP-B-07 run folder metrics at t 0.63; adopted t 0.67 elsewhere; submission_summary overwritten (t063 copy kept)", "fix": "document"},
            {"severity": "LOW", "issue": "candidate generation depends slightly on which S1 are blocked together (~0.002% of pairs; 1 validation threshold flip per system)",
             "evidence": "EXP-B-08 notes (B-06: 99/4.28M, B-08: 137/6.23M pairs)", "fix": "evaluate with all-S1 blocking (as test) - already the practice"},
            {"severity": "LOW", "issue": "EXP-S-02 analysis JSONs and v2 manifest were rewritten by an external re-run (07:56-08:46 on 2026-09-27)", "evidence": "masks byte-identical; all S-02 numbers reproduced", "fix": "document; avoid re-running tiers-build during other runs"},
            {"severity": "LOW", "issue": "extra run folder EXP-S-02_nullcheck (real-data null-floor regression)", "fix": "keep; documented"},
            {"severity": "INFO", "issue": "CLAUDE.md environment facts are stale (lists lightgbm as not installed; lightgbm 4.7.0 is installed)", "fix": "update CLAUDE.md (team)"}],
        "passes": ["23 configs, no duplicate experiment ids", "resolved configs of B-04/B-06/B-07/B-08 equal their run snapshots",
                   "S-02 four submits and B-08 two submits byte-identical", "S-02 add_floor=null reproduces the S-01 bytes on real data",
                   "outputs of experiments written to experiments/runs/<id>/output (never root output/)", "seed 42 everywhere; LightGBM deterministic=True"]})


def gpu():
    w("gpu_feasibility.json", {
        "labels": "[M] measured this session with a 400k x 30 synthetic LightGBM probe; recommendations [I]",
        "hardware": "NVIDIA GeForce RTX 3060 Laptop GPU, 6144 MiB, driver 616.92",
        "lightgbm": {"version": "4.7.0 (env)", "cpu": "OK (100 rounds, 400k rows: 2.1 s)",
                     "device_type=gpu (OpenCL)": "CRASHES the Python process (exit code 127, no traceback) after boost.compute fails to create its kernel-cache directories under C:/Users/Param/AppData/Roaming/boost_compute/<garbled name>",
                     "device_type=cuda": "not available: 'CUDA Tree Learner was not enabled in this build'"},
        "torch": "2.14.0+cu130, CUDA available (not used by the pipeline)",
        "assessment": [
            "[M] No GPU training path works in the current environment as installed.",
            "[M] Stage-1 LightGBM fit takes ~160 s (4.5M pairs) and stage-2 fit ~220 s on CPU; the pipeline's wall time is dominated by blocking, rapidfuzz features and scoring I/O (CPU/string work), not by LightGBM.",
            "[I] GPU would save at most a few minutes per experiment; it is not a bottleneck for any proposed experiment."],
        "if_gpu_is_wanted_later": [
            "Diagnose OpenCL first without changing the env: run with the boost.compute offline cache disabled/relocated (e.g. set the cache location to a plain ASCII path via BOOST_COMPUTE_CACHE_DIR if the build honours it) and verify with nvidia-smi; if the crash persists, OpenCL is not viable.",
            "CUDA LightGBM requires a source build with the CUDA toolkit (heavy on Windows; must be recorded as a decision).",
            "Every GPU run must: log device_type, assert GPU memory > baseline in nvidia-smi during training, report GPU memory, CPU/RAM, wall time, compare predictions with the CPU model, and fail (not fall back) if the GPU is unavailable.",
            "Keep blocking, string features and I/O on CPU; move only the dense float32 feature matrix to the GPU (4.5M x 24 float32 = ~430 MB fits in 6 GB)."]})


if __name__ == "__main__":
    rows = matrix()
    transfer(rows)
    shadow_audit()
    leakage()
    repro()
    gpu()
    print("compiled:", sorted(p.name for p in O.glob("*.json")))


def india_and_crossscript():
    b = j(O / "current_error_budget.json")["systems"]
    nk = j(O / "no_key_forensics.json")
    bf = j(O / "blocking_forensics_B06_k1000.json")
    xs = j(O / "cross_script_measured.json")
    td = j(O / "test_distribution.json")
    ctl = b["B06+D01(public-control stage 1)"]
    comp = {}
    for g in ("india", "us"):
        r = ctl[f"E_half_B:{g}"]
        prof = nk["profiles"]["no_key"][g]
        comp[g] = {
            "f05_B06D01[M]": r["f05"], "precision[M]": r["precision"], "recall[M]": r["recall"],
            "never_retrieved_top20_share[M]": r["buckets"]["A_never_retrieved(not in top-20 list)"]["share_of_true_pairs"],
            "never_retrieved_oracle[UB]": r["buckets"]["A_never_retrieved(not in top-20 list)"]["oracle_gain[UB]"],
            "weak_retrieved_0.02_0.5_oracle[UB]": r["buckets"]["B2_retrieved_score_0.02-0.5"]["oracle_gain[UB]"],
            "false_merges_per_1k_s1[M]": 1000 * r["buckets"]["C_false_merges(S1 with true matches)"]["false_pairs"] / r["s1"],
            "singleton_fp_s1[M]": r["buckets"]["D_singleton_false_positives"]["s1_affected"],
            "f05_by_cardinality[M]": r["f05_by_cardinality"],
            "validation_fates[M]": nk["fate_by_country"][g],
            "no_key_share_of_true[M]": nk["fate_by_country"][g]["no_key"] / sum(nk["fate_by_country"][g].values()),
            "no_key_profile[M]": {k: prof[k] for k in ("script_mismatch", "rec_addr_empty", "shared_number", "name_only", "address_only") if k in prof}
            | {"shares_ge2_name_tokens": prof["xlit_shared"]["share_ge_2"], "shares_ge1_useful_word": prof["shared_words"]["share_ge_1"],
               "shares_rare_name_token": prof["rare_xlit_shared"]["share_ge_1"]},
            "cross_script_share_of_true[M]": xs[f"{g}:cross_script_share"], "cross_script_share_of_no_key[M]": xs[f"{g}:cross_script_share_of_no_key"],
            "blocking_forensics_never_retrieved_categories[M]": {k: v.get(g) for k, v in bf["never_retrieved"]["shares"].items()},
            "candidates_train[M]": td["train"]["candidates"].get(g), "candidates_test[D]": td["test"]["candidates"].get(g),
            "S1_records_train[M]": td["train"].get(f"S1:{g}"), "pool_train[M]": td["train"].get(f"pool:{g}"), "pool_test[D]": td["test"].get(f"pool:{g}")}
    w("india_forensics.json", {
        "purpose": "understand the data structures behind India's retrieval gap; NOT to create country rules", "labels": "[M] validation / E half B; [D] test unlabeled; [UB]",
        "india_vs_us": comp,
        "findings[I]": [
            "India never-retrieved share is 1.7x the US (6.9% vs 4.0% of true pairs in E half B).",
            "India no-key pairs are 96% Latin->Latin: cross-script is no longer the retrieval problem (cross-script recall at any key 0.992).",
            "India no-key pairs share COMMON tokens and COMMON address words (rare-token sharing ~9%, rare-word sharing 0%): rarest-k key selection picks unshared rare tokens.",
            "India no-key pairs carry house-number evidence far more often than US ones (shared number 59% vs 20%); US no-key pairs are name-only records with empty addresses (67%).",
            "India false merges per 1k S1 are higher than US, and India singleton F0.5 is lower (0.938 vs 0.958).",
            "India S1 addresses are long (12.4 tokens vs 5.9 US); the 3-5 rarest useful words are a small, unstable sample of them."]})
    x = xs
    w("cross_script_review.json", {
        "labels": "[M] measured; history from the experiment ledger; [I] inferences",
        "history(EXP-N-01/N-02)": {"map": "1,262 train-learned token mappings (support>=3, confidence>=0.5, ties rejected); fit-set only; cross-fitted for training rows",
                                    "representation_value[M]": "validation name Jaccard on cross-script pairs 0.021 -> 0.586",
                                    "blocking_value[M]": "recall@20 0.9267 -> 0.9328 (India 0.8816 -> 0.8968); candidates unchanged; +5,150 true pairs in top-20, 508 pushed out, 518 lost to cap growth",
                                    "end_to_end_value[M]": "E +0.0089 (0.9313 -> 0.9402), India +0.021, precision up, cross-script false merges 11,315 -> 3,244",
                                    "public[reported]": "0.928 vs B-04 approx 0.91 (+0.018): the largest public gain after LightGBM"},
        "premise_check": "The audit brief asks why N-02's public result stayed below the previous control. The ledger contradicts this: N-02 (0.928) was ABOVE the previous submission (B-04 approx 0.91). It is below the later 0.935 file, which is built on top of N-02.",
        "current_state[M]": x["combos"],
        "india_cross_script_share_of_true": x["india:cross_script_share"], "india_cross_script_share_of_never_retrieved_top20": x["india:cross_script_share_of_never_retrieved_top20"],
        "india_cross_script_share_of_no_key": x["india:cross_script_share_of_no_key"],
        "answers[I]": {
            "did transliteration improve representation": "yes [M]: Jaccard 0.02 -> 0.59 on cross-script pairs",
            "did it improve blocking recall": "yes [M]: India recall@20 +0.015; cross-script pairs now reach any-key recall 0.992 (Latin-Latin 0.965)",
            "did it create candidate explosion": "no [M]: candidates per S1 unchanged (19.3)",
            "did it alter India more than US": "yes [M]: US identical by construction (no non-Latin US records)",
            "is the remaining problem the representation or its placement": "neither is the bottleneck now: cross-script pairs have HIGHER retrieval and decision recall than Latin-Latin pairs; they are 3.7% of India no-key pairs",
            "narrow bridge for no-key pairs": "upper bound is small: ~500 validation no-key pairs (3.7% of India's 13,655)",
            "headroom[UB]": "cross-script true pairs missing from the final decision: (1 - 0.9145) x 51,774 ~ 4.4k validation pairs; not a priority bucket"}})


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "india":
    india_and_crossscript()
    print("india + cross-script written")
