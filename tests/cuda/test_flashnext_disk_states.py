"""Flash Next's kept prompt states on disk on the GPU: a state read back equals the one kept in memory bit for bit, and
a next turn resumed from disk replies as the same turn resumed from memory and as a cold prefill, greedy and sampled,
on the serial engine and the concurrent decoder (bf16 and int8 KV)."""

import threading
import time

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import _model  # noqa: E402
from test_flashnext_prompt_cache import (  # noqa: E402
    CHUNK, END_THINK, NL, NL2, THINK, _engine_shim, _generate, _prompt, _same_bits, _same_snap)

from tensorfold.cuda import disk_states  # noqa: E402
from tensorfold.cuda.disk_states import DiskStates  # noqa: E402
from tensorfold.cuda.scheduler import Scheduler  # noqa: E402
from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import disk  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402

SAMPLINGS = [None, Sampling(seed=4321, top_k=20, top_p=0.95)]


@pytest.fixture(scope="module")
def w():
    return _model()


@pytest.fixture(autouse=True)
def short_states(monkeypatch):
    """Toy prompts of a few hundred tokens: states that short go to disk too."""

    monkeypatch.setattr(disk_states, "MIN_TOKENS", 64)


def _turns(base: int = 300):
    head = _prompt(base, seed=base)
    first = head + [THINK, NL]
    second = head + [THINK, NL2, END_THINK, NL2] + _prompt(7, seed=6) + [THINK, NL]
    return first, second


def _same_rows(a, b, n: int, m: int) -> bool:
    return all(_same_bits(x, y) for x, y in zip(disk.rows(a, n, m), disk.rows(b, n, m)))


def _cold(w, prompt, sampling, count=12) -> list[int]:
    e = Engine(w, capacity=1024, max_rows=8, prefill_rows=CHUNK)
    first = prefill(e, prompt, sampling)
    return serial_decode(e, first, count, sampling, stop_eos=False).tokens


@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_the_serial_engine_resumes_from_disk_as_from_memory(w, tmp_path, sampling):
    first, second = _turns()
    kept = _engine_shim(w)
    kept.disk = DiskStates(tmp_path, "k", 1 << 30)
    _generate(kept, first, sampling)
    ids, mem = kept.cache[-1]
    assert kept.persist(lambda: False) >= 1 and kept.disk.has(ids)

    restarted = _engine_shim(w)                                  # nothing in memory, the same folder
    restarted.disk = DiskStates(tmp_path, "k", 1 << 30)
    entry = restarted.disk.find(second, 0, lambda e: True)
    assert entry.ids == ids
    hit = restarted._load_disk(entry)
    n, m = len(ids), mem["state"]["mtp_len"]
    assert _same_snap(hit[1]["state"], mem["state"]) and _same_bits(hit[1]["tail"], mem["tail"])
    assert _same_rows(restarted.e.st, kept.e.st, n, m)

    restarted = _engine_shim(w)
    restarted.disk = DiskStates(tmp_path, "k", 1 << 30)
    from_disk, stats = _generate(restarted, second, sampling)
    assert stats["disk_tokens"] == n and stats["cached"] == n
    from_memory, stats = _generate(kept, second, sampling)
    assert stats["cached"] == n and "disk_tokens" not in stats
    assert from_disk == from_memory == _generate(_engine_shim(w), second, sampling)[0]
    assert from_disk == _generate(_engine_shim(w), second, sampling, draft=False)[0]


def _decoder(w, kv: str, path) -> MultiDecoder:
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv)
    dec.disk = DiskStates(path, "k", 1 << 30)
    return dec


def _run(dec, prompt, sampling, count=12):
    got: list[int] = []
    s = Stream(list(prompt), count, sampling, emit=lambda new: got.extend(new))
    dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    return s, got


@pytest.mark.parametrize("kv", ["bf16", "int8"])
@pytest.mark.parametrize("sampling", SAMPLINGS)
def test_the_concurrent_decoder_resumes_from_disk_as_from_memory(w, tmp_path, kv, sampling):
    first, second = _turns()
    kept = _decoder(w, kv, tmp_path)
    _run(kept, first, sampling)
    (ids, slot, snap, tail), = [k for k in kept.kept if k[0] == first[:-1]]
    assert kept.persist(lambda: False) >= 1

    restarted = _decoder(w, kv, tmp_path)
    s, from_disk = _run(restarted, second, sampling)
    assert s.disk_tokens == len(ids) and s.cached == len(ids)
    # the state read back, kept in memory again, is the one written
    (_, slot2, snap2, tail2), = [k for k in restarted.kept if k[0] == ids]
    assert _same_snap(snap2, snap) and _same_bits(tail2, tail)
    assert _same_rows(slot2, slot, len(ids), snap["mtp_len"])
    s, from_memory = _run(kept, second, sampling)
    assert s.cached == len(ids) and s.disk_tokens == 0
    s, cold = _run(_decoder(w, kv, tmp_path / "none"), second, sampling)
    assert s.cached == 0
    assert from_disk == from_memory == cold
    if kv == "bf16":
        assert from_disk == _cold(w, second, sampling)


def test_the_scheduler_writes_after_a_length_finish_and_a_restart_reads_it(w, tmp_path):
    first, second = _turns()
    dec = _decoder(w, "int8", tmp_path)
    sched = Scheduler(dec, max_streams=2, idle=dec.persist)
    try:
        sched.submit(first, 3, None, True, lambda new: False)    # ends at its length
        deadline = time.monotonic() + 30
        while not dec.disk.has(first[:-1]) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert dec.disk.has(first[:-1])
        assert sched.call(lambda: threading.current_thread()) is sched.thread
    finally:
        sched.close()
    restarted = _decoder(w, "int8", tmp_path)
    s, got = _run(restarted, second, None)
    assert s.disk_tokens == len(first) - 1
    s, want = _run(_decoder(w, "int8", tmp_path / "none"), second, None)
    assert got == want
