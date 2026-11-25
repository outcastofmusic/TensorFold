"""The CUDA engines' tests: collected only where PyTorch sees an NVIDIA GPU (DGX Spark, in NVIDIA's container).

WHOLE_MODEL lists the modules that load the 95 GB GLM checkpoint. Each runs only when named on the command line, in
a process of its own: after the rest of the suite one process still holds about 3 GiB of GPU memory and some host
memory, and GLM's admission then finds no room for its 8,192-token window (25 Nov). So ``pytest tests/cuda`` and then
``pytest tests/cuda/<module>`` for each of them.
"""

import gc
import importlib.util
import os

import pytest

# GLM's engines keep the MTP head beside DFlash2 here (TF_GLM_MTP=1), so both drafters stay under test
os.environ.setdefault("TF_GLM_MTP", "1")


def _cuda() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available()


collect_ignore_glob = [] if _cuda() else ["test_*.py"]

WHOLE_MODEL = ("test_glm_exl3_replies.py", "test_glm_packed_in_place.py", "test_glm_sweep_switches.py")


def pytest_ignore_collect(collection_path, config):
    """A WHOLE_MODEL module is collected only when the command line names it."""

    if collection_path.name in WHOLE_MODEL and not any(collection_path.name in str(a) for a in config.args):
        return True
    return None


def pytest_report_header(config):
    return f"whole-model modules, collected only when named: {', '.join(WHOLE_MODEL)}"


def _available_gib() -> float:
    """MemAvailable: the Spark's GPU and CPU share it, and engine admission sizes from it."""

    with open("/proc/meminfo") as f:
        return next(int(line.split()[1]) for line in f if line.startswith("MemAvailable")) / 2**20


@pytest.fixture(autouse=True, scope="module")
def _give_back_gpu_memory(request):
    """After each module: free its tensors and torch's cached blocks, so the next engine admits as a fresh process.
    A module that still holds over 1 GiB is named on stderr: the next whole-model module will not fit beside it."""

    yield
    if not _cuda():
        return
    import sys

    import torch

    if torch.cuda.is_initialized():
        gc.collect()
        torch.cuda.empty_cache()
        held = torch.cuda.memory_allocated() / 2**30
        avail = _available_gib()
        line = f"[gpu memory] after {request.module.__name__}: {held:.1f} GiB allocated, {avail:.0f} GiB available"
        if os.environ.get("TF_TEST_MEMORY_LOG"):                 # every module, for finding what holds memory
            with open(os.environ["TF_TEST_MEMORY_LOG"], "a") as f:
                print(line, file=f)
        elif held > 1 or avail < 90:
            print("\n" + line, file=sys.stderr)
