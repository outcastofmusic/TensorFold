"""The CPU EXL3 tests' fakes: a recording extension, a small routed layer, and ptxas's verdict on sm_121 instances."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import torch

from tensorfold.cuda.exl3 import experts


class FakeExt:
    """Records each launch's name and its trailing switches (keyword arguments in ``kwargs``, by launch)."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.kwargs: list[dict] = []

    def __getattr__(self, name):
        def call(*args, **kw):
            self.calls.append((name, args))
            self.kwargs.append(kw)
        return call


def _layer(D=256, I=128, E=8):
    t = torch.zeros(E, dtype=torch.int64)
    k2 = torch.full((E,), 6, dtype=torch.int32)
    h = torch.zeros((E, D), dtype=torch.float16)
    return experts.Exl3RoutedExperts(t, t, t, k2, k2, k2, h, h, h[:, :I], h[:, :I], h[:, :I], h, E, D, I,
                                     experts.CB_MUL1, (6, 6), (6, 6), t)


HERE = Path(experts.__file__).parent
NVCC = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)


def _ptxas(kernels: list[str]) -> dict[str, tuple[int, int, bool]]:
    """{instance: (registers, shared bytes, spills)} of the named grouped_kernel instances, built for sm_121."""

    from torch.utils import cpp_extension

    # grouped_kernel's operands, then its item list, the list's count and the epilogue
    src = "#include \"experts_grouped.cuh\"\nnamespace tf_exl3x {\n" + "".join(
        f"template __global__ void {k}(const half*, const half*, const int64_t*, const int64_t*, const int*, "
        "const int*, const int*, const int*, const int*, float*, int, int, int, int, int, int, uint32_t, "
        "const int*, const int*, Epi);\n"
        for k in kernels) + "}\n"
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        cu = Path(tmp) / "inst.cu"
        cu.write_text(src)
        cmd = [NVCC, "-O3", "-std=c++17", "-gencode=arch=compute_121,code=sm_121", "-Xptxas", "-v", "-c", str(cu),
               "-o", str(Path(tmp) / "inst.o"), f"-I{HERE}"] + [f"-I{p}" for p in cpp_extension.include_paths()]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    got, name = {}, None
    for line in out.stderr.splitlines():
        m = re.search(r"Function properties for (\S+)", line)
        if m:
            name = m.group(1)
            continue
        if name and "spill" in line:
            spills = not re.search(r"0 bytes spill stores, 0 bytes spill loads", line)
            got[name] = [0, 0, spills]
        m = re.search(r"Used (\d+) registers.*?(\d+) bytes smem", line)
        if name and m:
            got[name][0], got[name][1] = int(m.group(1)), int(m.group(2))
            name = None
    return {k: tuple(v) for k, v in got.items()}
