"""Stage-1 scores for ALL train S1 under full competition (the stage-2 training input; out of sample for every
S1 outside the stage-1 fit set). Thin CLI around ``score_split`` -- the same call the EXP-B-09 analysis used to
build ``forensic_alltrain_B09``.

    python -m ber.pipeline.score_alltrain --config configs/exp_b09_name_num.yaml --tag forensic_alltrain_B09
"""

from __future__ import annotations

import argparse

from ber.config import Paths, load_config
from ber.models import make_scorer
from ber.pipeline.run_baseline import score_split
from ber.utils.runtime import LOG, PeakMemory, configure_arrow, setup_logging, stage


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", required=True, help="scored-parts tag, e.g. forensic_alltrain_B09 (= stage2.train_tag)")
    a = ap.parse_args(argv)
    setup_logging()
    configure_arrow()
    cfg = load_config(a.config)
    paths = Paths(cfg)
    out = paths.scored_dir("train", a.tag)
    with stage("score ALL train S1"), PeakMemory() as pm:
        st = score_split(cfg, paths, "train", None, out, make_scorer(cfg, paths), store_features=False)
    LOG.info("all-train scoring -> %s | pairs %s | S1 %s | peak %.2f GB", out.as_posix(), st.get("candidate_pairs"), st.get("evaluated_s1"), pm.peak_gb)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
