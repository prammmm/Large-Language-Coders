# Business Entity Resolution — LargeLanguageCoders

Python 3.13, CPU only. No external data or pretrained models; LightGBM (MIT) is the only model.

## Setup

```bash
pip install -r requirements.txt
export PYTHONPATH=src            # PowerShell: $env:PYTHONPATH="src"
```

Place the official data in `datasets/train/` and `datasets/test/`. Run every command from this folder.

## Full reproduction

```bash
# 1. data
python -m ber.pipeline.run_baseline preprocess --config configs/exp_b09_name_num.yaml
python -m ber.pipeline.run_baseline split      --config configs/exp_b09_name_num.yaml

# 2. pipeline A: blocker B-09 + stage-1 LightGBM + stage-2 veto
python -m ber.pipeline.run_baseline fit   --config configs/exp_b09_name_num.yaml
python -m ber.pipeline.run_baseline infer --config configs/exp_b09_name_num.yaml
python -m ber.pipeline.score_alltrain --config configs/exp_b09_name_num.yaml --tag forensic_alltrain_B09
python -m ber.pipeline.run_stage2 features --config configs/exp_s03_veto_on_b09.yaml
python -m ber.pipeline.run_stage2 fit      --config configs/exp_s03_veto_on_b09.yaml

# 3. pipeline B: blocker B-10 + stage-1 LightGBM + stage-2 veto
python -m ber.pipeline.run_baseline fit   --config configs/exp_s04_stage1_b10.yaml
python -m ber.pipeline.run_baseline infer --config configs/exp_s04_stage1_b10.yaml
python -m ber.pipeline.score_alltrain --config configs/exp_s04_stage1_b10.yaml --tag forensic_alltrain_S04
python -m ber.pipeline.run_stage2 features --config configs/exp_s04_veto_on_b10.yaml
python -m ber.pipeline.run_stage2 fit      --config configs/exp_s04_veto_on_b10.yaml

# 4. pipeline C: blocker with top-30 candidates + stage-1 LightGBM
python -m ber.pipeline.run_baseline fit   --config configs/exp_b08_topk30.yaml
python -m ber.pipeline.run_baseline infer --config configs/exp_b08_topk30.yaml

# 5. final combination -> output/matching_results.tsv, output/candidate_pairs.tsv
python -m ber.pipeline.final_submit --config configs/final.yaml
```

Steps 3–4 can also run in any order after step 1. Peak memory is about 7.5 GB (step 5); every other step is at most 5 GB.

## Quick path (trained models included)

The trained models are in `artifacts/`. You can skip every `fit` and `score_alltrain` command, and run the two `features` commands with `--split test`:

```bash
python -m ber.pipeline.run_baseline preprocess --config configs/exp_b09_name_num.yaml
python -m ber.pipeline.run_baseline split      --config configs/exp_b09_name_num.yaml
python -m ber.pipeline.run_baseline infer --config configs/exp_b09_name_num.yaml
python -m ber.pipeline.run_baseline infer --config configs/exp_s04_stage1_b10.yaml
python -m ber.pipeline.run_baseline infer --config configs/exp_b08_topk30.yaml
python -m ber.pipeline.run_stage2 features --split test --config configs/exp_s03_veto_on_b09.yaml
python -m ber.pipeline.run_stage2 features --split test --config configs/exp_s04_veto_on_b10.yaml
python -m ber.pipeline.final_submit --config configs/final.yaml
```

The output is byte-identical to `../../output/`.
