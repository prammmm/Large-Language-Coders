from __future__ import annotations
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
try:
    import psutil
except ImportError:
    psutil = None
LOG = logging.getLogger('ber')

def setup_logging(level: int=logging.INFO) -> None:
    if not LOG.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s', '%H:%M:%S'))
        LOG.addHandler(h)
    LOG.setLevel(level)

def configure_arrow() -> None:
    """Use the system allocator: Arrow's default (mimalloc) pool keeps freed memory, which"""
    import pyarrow as pa
    pa.set_memory_pool(pa.system_memory_pool())

def rss_gb() -> float:
    if psutil is None:
        return float('nan')
    return psutil.Process().memory_info().rss / 1000000000.0

def available_gb() -> float:
    if psutil is None:
        return float('nan')
    return psutil.virtual_memory().available / 1000000000.0

@contextmanager
def stage(name: str):
    t0 = time.perf_counter()
    LOG.info('[%s] start (rss %.2f GB, available %.2f GB)', name, rss_gb(), available_gb())
    yield
    LOG.info('[%s] done in %.1fs (rss %.2f GB)', name, time.perf_counter() - t0, rss_gb())

class PeakMemory:
    """Context manager sampling this process's RSS in a thread; ``.peak_gb`` afterwards."""

    def __init__(self, interval: float=0.1):
        self.interval = interval
        self.peak_gb = float('nan')

    def __enter__(self):
        import threading
        self._stop = threading.Event()
        self._peak = 0

        def run():
            while not self._stop.is_set():
                if psutil is not None:
                    self._peak = max(self._peak, psutil.Process().memory_info().rss)
                self._stop.wait(self.interval)
        self._t = threading.Thread(target=run, daemon=True)
        self._t.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.seconds = time.perf_counter() - self._t0
        self.peak_gb = self._peak / 1000000000.0 if self._peak else float('nan')
        return False

def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f'not JSON serialisable: {type(o)}')

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(obj, f, indent=2, default=_json_default, ensure_ascii=False)
        f.write('\n')

def read_json(path: Path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)
