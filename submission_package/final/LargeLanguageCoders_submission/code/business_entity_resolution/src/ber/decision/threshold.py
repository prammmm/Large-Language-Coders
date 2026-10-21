from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from ber.utils.runtime import LOG, read_json

def apply_global_threshold(score: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(score) >= np.float32(threshold)

def threshold_source(decision_cfg: dict) -> str:
    src = decision_cfg.get('threshold_from')
    if src is None:
        src = 'val_sweep' if decision_cfg.get('use_tuned_threshold', False) else 'config'
    if src not in ('val_sweep', 'model', 'config'):
        raise ValueError(f'decision.threshold_from must be val_sweep|model|config, got {src!r}')
    return src

def resolve_threshold(decision_cfg: dict, tuned_path: Path, model_file: Path | None=None) -> tuple[float, str]:
    src = threshold_source(decision_cfg)
    if src == 'val_sweep' and tuned_path.exists():
        t = float(read_json(tuned_path)['threshold'])
        LOG.info('using validation-tuned threshold %.4f from %s', t, tuned_path)
        return (t, str(tuned_path))
    if src == 'model' and model_file is not None and Path(model_file).exists():
        meta = json.loads(Path(model_file).read_text(encoding='utf-8')).get('meta', {})
        if 'threshold' in meta:
            LOG.info('using train-fold (inner split) threshold %.4f from %s', meta['threshold'], model_file)
            return (float(meta['threshold']), f'{model_file} (inner train-fold split)')
    t = float(decision_cfg['threshold'])
    if src != 'config':
        LOG.warning('threshold source %r unavailable; falling back to config threshold %.4f', src, t)
    return (t, 'config')
