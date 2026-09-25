"""Environment check: python/dependency versions, required imports, optional CUDA info.

    python scripts/verify_env.py        (exit 0 = everything required is importable)
"""

from __future__ import annotations

import importlib
import platform
import sys

REQUIRED = ["numpy", "pandas", "pyarrow", "rapidfuzz", "yaml", "pytest", "psutil", "ber"]
BER_MODULES = [
    "ber.normalization.views",
    "ber.data.ingest",
    "ber.data.normalize_records",
    "ber.data.split",
    "ber.blocking.blocker",
    "ber.features.pair_features",
    "ber.models.weighted_rule",
    "ber.decision.threshold",
    "ber.evaluation.metrics",
    "ber.submission.writer",
    "ber.submission.strict_validator",
    "ber.pipeline.run_baseline",
]


def main() -> int:
    ok = True
    print(f"python      {sys.version.split()[0]}  ({sys.executable})")
    if "ber_hackathon" not in sys.executable:
        print(
            "WARNING     this is NOT the ber_hackathon env's python. Run `conda activate ber_hackathon` first\n"
            "            (PowerShell needs a one-time `conda init powershell` + new terminal), or call\n"
            "            E:\\hell\\hell\\envs\\ber_hackathon\\python.exe directly."
        )
    print(f"platform    {platform.platform()}")
    for name in REQUIRED:
        try:
            m = importlib.import_module(name)
            print(f"{name:<11} {getattr(m, '__version__', 'ok')}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"{name:<11} MISSING ({e})")
    for name in BER_MODULES:
        try:
            importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"import FAILED: {name}: {e}")
    try:
        import psutil

        vm = psutil.virtual_memory()
        print(f"memory      total {vm.total / 1e9:.1f} GB, available {vm.available / 1e9:.1f} GB, cpus {psutil.cpu_count()}")
    except Exception:  # noqa: BLE001
        pass
    try:  # optional: not needed for the Submission-1 baseline
        import torch

        cuda = torch.cuda.is_available()
        print(f"torch       {torch.__version__}  cuda={cuda}" + (f"  gpu={torch.cuda.get_device_name(0)}" if cuda else ""))
    except ImportError:
        print("torch       not installed (not required for the baseline)")
    print("RESULT      OK" if ok else "RESULT      FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
