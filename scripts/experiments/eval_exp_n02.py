"""EXP-N-02 evaluation: B-04 + D-01 (control) vs N-02 + D-01 under FULL competition.

Same two-arm evaluator as EXP-B-05 (``eval_exp_b05.py``: E = all train S1 not in the shared 200k
fit set, all 2,206,821 train S1 compete, D-01 = production ``best_claimant_keep``), with the
cross-script (script-mismatch: Latin S1 name vs non-Latin / mixed pool name) accounting switched on,
on train (labelled) and on test (unlabelled diagnostics).

    python scripts/experiments/eval_exp_n02.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_exp_b05  # noqa: E402

DEFAULTS = ["--exp", "configs/exp_n02_transliteration.yaml", "--exp-tag", "forensic_alltrain_N02", "--label", "N02",
            "--out", "experiments/runs/EXP-N-02/full_competition_comparison.json", "--script-analysis"]

if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + DEFAULTS + sys.argv[1:]
    eval_exp_b05.main()
