import sys
from contextlib import contextmanager

FLAT = ("config", "common", "model", "predict", "metrics", "dataset", "censor", "align", "runs",
        "calibrate", "baseline", "normalize", "train", "evaluate")


@contextmanager
def task_scope(task_dir):
    """folder task -> sys.path dan sys.modules sementara"""
    saved_path = list(sys.path)
    saved = {n: sys.modules.pop(n) for n in FLAT if n in sys.modules}
    sys.path.insert(0, str(task_dir))
    try:
        yield
    finally:
        # modul task tetap hidup lewat referensi objek, namanya dilepas dari sys.modules
        for name in FLAT:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = saved_path
