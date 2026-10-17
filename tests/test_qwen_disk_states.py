"""Flash Next's kept prompt states on disk (``qwen4_exp.cuda.disk``) without a GPU: a state's rows and snapshot read
back bit for bit into another slot, the concurrent decoder and the serial engine resume from a disk state only when
it is longer than any kept in memory, and a piece that fails its hash prefills instead."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.torch

ATT, HEADS, DIM, INDEX, RATIO = 3, 2, 64, 16, 4


def slot(capacity=1400, kv="int8", mtp=True):
    """A slot's caches as ``State`` holds them: KV caches, index keys, pools, the MTP head's, on the CPU."""

    import torch

    from tensorfold.families.qwen4_exp.cuda.kvcache import KVCache

    st = SimpleNamespace(ratio=RATIO, image_positions=None, capacity=capacity,
                         kc=[KVCache(capacity, HEADS, DIM, "cpu", kv) for _ in range(ATT)],
                         ikc=[torch.zeros(capacity, INDEX, dtype=torch.bfloat16) for _ in range(ATT)],
                         pooled=[torch.zeros(capacity // RATIO, INDEX, dtype=torch.bfloat16) for _ in range(ATT)],
                         rec=torch.zeros(2, 5, 3, 4, 4), conv=torch.zeros(5, 3, 8, dtype=torch.bfloat16),
                         ple_tail=torch.zeros(9, 8, dtype=torch.bfloat16), ple_history=np.zeros(2, dtype=np.int64))
    if mtp:
        st.mtp_kc = KVCache(capacity, HEADS, DIM, "cpu", kv)
        st.mtp_ikc = torch.zeros(capacity, INDEX, dtype=torch.bfloat16)
        st.mtp_pooled = torch.zeros(capacity // RATIO, INDEX, dtype=torch.bfloat16)
    return st


def noise(st, seed=0):
    """Random bytes in every cache row (NaN patterns included: the checks compare bytes)."""

    import torch

    from tensorfold.families.qwen4_exp.cuda.disk import rows

    g = torch.Generator().manual_seed(seed)
    for v in rows(st, st.capacity, st.capacity):
        raw = v.view(torch.uint8)
        raw.copy_(torch.randint(0, 256, raw.shape, generator=g, dtype=torch.uint8))
    return st


def kept(n, seed=0, tail=True):
    """A kept state's snapshot and tail, as ``prefill_chunk`` keeps them."""

    import torch

    g = torch.Generator().manual_seed(seed)
    snap = {"pos": n, "rec": torch.randn(5, 3, 4, 4, generator=g),
            "conv": torch.randn(5, 3, 8, generator=g).to(torch.bfloat16),
            "ple_tail": torch.randn(9, 8, generator=g).to(torch.bfloat16),
            "ple_history": np.array([7 + seed, 11], dtype=np.int64), "mtp_len": n - 1}
    return snap, (torch.randn(1, 8, generator=g).to(torch.bfloat16) if tail else None)


def same(a, b) -> bool:
    import torch

    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.contiguous().view(torch.uint8),
                                                                     b.contiguous().view(torch.uint8))


def store(tmp_path, budget=1 << 30):
    from tensorfold.cuda.disk_states import DiskStates

    return DiskStates(tmp_path, "k", budget)


def same_resume(got, snap, tail) -> bool:
    s = got["state"]
    return (s["pos"] == snap["pos"] and s["mtp_len"] == snap["mtp_len"]
            and all(same(s[k], snap[k]) for k in ("rec", "conv", "ple_tail"))
            and np.array_equal(s["ple_history"], snap["ple_history"])
            and s["ple_history"].dtype == snap["ple_history"].dtype
            and (got["tail"] is None if tail is None else same(got["tail"], tail)))


@pytest.mark.parametrize("kv", ["int8", "bf16", "int4"])
def test_a_state_reads_back_into_another_slot_bit_for_bit(tmp_path, kv):
    from tensorfold.families.qwen4_exp.cuda import disk

    src, n = noise(slot(kv=kv)), 1300
    snap, tail = kept(n)
    assert disk.persist(store(tmp_path), [(list(range(n)), src, snap, tail)], lambda: False) == 1
    again = store(tmp_path)                                       # a restart finds it
    entry = again.find([*range(n), 5], 0, lambda e: True)
    dst = slot(kv=kv)
    got = disk.load(again, entry, dst)
    assert got is not None and same_resume(got, snap, tail)
    assert all(same(a, b) for a, b in zip(disk.rows(dst, n, n - 1), disk.rows(src, n, n - 1)))
    # the rows past the state and the partial pool block are not part of it
    assert not dst.kc[0].k[n:].any() and not dst.pooled[0][n // RATIO:].any() and not dst.mtp_kc.k[n - 1:].any()


def test_the_rows_are_what_a_resume_reads(tmp_path):
    from tensorfold.families.qwen4_exp.cuda import disk

    st = slot(kv="int8")
    got = disk.rows(st, 1025, 1024)
    # each attention layer: keys, values, their scales, index keys, complete pools; then the MTP head's the same
    assert len(got) == 6 * (ATT + 1)
    assert [v.shape[0] for v in got[:6]] == [1025] * 5 + [1025 // RATIO]
    assert [v.shape[0] for v in got[-6:]] == [1024] * 5 + [1024 // RATIO]
    assert len(disk.rows(slot(kv="bf16"), 10, 9)) == 4 * (ATT + 1)    # a bf16 cache's one-element scales are no rows
    assert len(disk.rows(slot(mtp=False), 10, 9)) == 6 * ATT


def test_short_states_and_states_on_disk_are_not_written_again(tmp_path):
    from tensorfold.cuda.disk_states import MIN_TOKENS
    from tensorfold.families.qwen4_exp.cuda import disk

    st, d = noise(slot()), store(tmp_path)
    short, long = MIN_TOKENS - 1, MIN_TOKENS + 100
    entries = [(list(range(short)), st, *kept(short)), (list(range(long)), st, *kept(long))]
    assert disk.persist(d, entries, lambda: False) == 1
    assert disk.persist(d, entries, lambda: False) == 0
    assert [len(e.ids) for e in d.entries] == [long]


def test_a_stop_ends_the_pass_and_the_next_one_writes_the_rest(tmp_path):
    from tensorfold.families.qwen4_exp.cuda import disk

    st, d = noise(slot()), store(tmp_path)
    entries = [(list(range(1100)), st, *kept(1100)), (list(range(1300)), st, *kept(1300))]
    asked = iter([False, False, True])
    assert disk.persist(d, entries, lambda: next(asked, True)) == 0
    assert d.entries == []
    assert disk.persist(d, entries, lambda: False) == 2


def test_a_bad_piece_reads_as_no_state_and_is_forgotten(tmp_path, capsys):
    from tensorfold.families.qwen4_exp.cuda import disk

    d, n = store(tmp_path), 1100
    disk.persist(d, [(list(range(n)), noise(slot()), *kept(n))], lambda: False)
    entry = d.entries[0]
    victim = d.pieces / entry.meta["extras"]["rec"]["pieces"][0]
    victim.write_bytes(b"\1" + victim.read_bytes()[1:])
    assert disk.load(d, entry, slot()) is None
    assert "did not read" in capsys.readouterr().out
    assert d.find([*range(n), 1], 0, lambda e: True) is None


def test_the_fingerprint_holds_the_shards_the_toolchain_and_the_shared_code(tmp_path, monkeypatch):
    import os

    import torch

    from tensorfold.families.qwen4_exp.cuda import disk

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    shard = model / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"\0" * 64)
    monkeypatch.setenv("TENSORFOLD_QWEN_STATES", str(tmp_path / "states"))
    key = lambda: disk.open_states(model, slot(), engine="serial", prefill_rows=256).key   # noqa: E731
    first = key()
    assert key() == first
    os.utime(shard, ns=(1, 1))                                       # the same name, another shard
    second = key()
    (model / "model.safetensors.index.json").write_text('{"weight_map": {"x": "y"}}')
    third = key()
    monkeypatch.setattr(torch, "__version__", "0.0.0")
    fourth = key()
    assert len({first, second, third, fourth}) == 4
    code = disk._code()
    assert any(f.name == "scheduler.py" and f.parent.name == "cuda" for f in code)        # the shared package
    assert any(f.name == "decode.py" and f.parent.name == "qwen4_exp" for f in code)      # the whole family
    assert any(f.suffix == ".cu" for f in code)


def test_a_read_fills_each_view_from_one_buffer(tmp_path, monkeypatch):
    """Every view's pieces go into one buffer and one copy, and each small tensor's likewise, bit for bit."""

    from tensorfold.cuda import disk_states
    from tensorfold.families.qwen4_exp.cuda import disk

    monkeypatch.setattr(disk_states, "PIECE_BYTES", 500)             # rec (960 bytes) in two pieces
    d, n, src = store(tmp_path), 1300, noise(slot())
    snap, tail = kept(n)
    disk.persist(d, [(list(range(n)), src, snap, tail)], lambda: False)
    entry = d.entries[0]
    assert len(entry.meta["extras"]["rec"]["pieces"]) == 2 and len(entry.meta["views"][0]["pieces"]) == 6
    gathered = []
    real = d._gather
    monkeypatch.setattr(d, "_gather", lambda names, *a: gathered.append(len(names)) or real(names, *a))
    dst = slot()
    got = disk.load(d, entry, dst)
    assert same_resume(got, snap, tail)
    assert all(same(a, b) for a, b in zip(disk.rows(dst, n, n - 1), disk.rows(src, n, n - 1)))
    views = disk.rows(dst, n, n - 1)
    assert len(gathered) == len(views) + len(entry.meta["extras"])
    assert gathered[:len(views)] == [len(s["pieces"]) for s in entry.meta["views"]]


def test_a_piece_of_the_wrong_length_refuses_the_state(tmp_path):
    from tensorfold.families.qwen4_exp.cuda import disk

    d, n = store(tmp_path), 1100
    disk.persist(d, [(list(range(n)), noise(slot()), *kept(n))], lambda: False)
    entry = d.entries[0]
    pieces = entry.meta["views"][0]["pieces"]
    pieces[-1] = d._put((d.pieces / pieces[-1]).read_bytes() + b"\0")    # sound by its hash, a byte too long
    assert disk.load(d, entry, slot()) is None and d.entries == []


def test_a_full_disk_ends_the_pass_with_a_line(tmp_path, capsys, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import disk

    d = store(tmp_path)

    def full(data):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(d, "_put", full)
    assert disk.persist(d, [(list(range(1100)), noise(slot()), *kept(1100))], lambda: False) == 0
    assert "No space left" in capsys.readouterr().out


# -- the concurrent decoder ------------------------------------------------------------------------------
def decoder(monkeypatch, d, slots):
    """A bare ``MultiDecoder`` on host slots: admission runs as written, the prompt's first rows are recorded."""

    from tensorfold.families.qwen4_exp.cuda import multi

    m = object.__new__(multi.MultiDecoder)
    m.w = SimpleNamespace(comm=None)
    m.capacity, m.depth, m.prefill_rows, m.points, m.vision = 100_000, 2, 256, None, None
    m.streams, m.filling, m.fills, m.kept, m.keep, m.free = {}, [], {}, [], 8, list(slots)
    m.solo, m.splices, m.next_id, m.disk = None, False, 0, d
    m.buf, m.mbuf, m.pbuf = SimpleNamespace(rows=3), None, None
    m._grow = lambda st, rows, **kw: True
    resumed = []

    def begin(e, prompt, *, mtp, resume):
        resumed.append(resume)
        return resume["state"]["pos"] if resume is not None else 0

    monkeypatch.setattr(multi, "prefill_begin", begin)
    monkeypatch.setattr(multi.image_rows, "begin", lambda *a: None)
    return m, resumed


def stream(prompt):
    from tensorfold.cuda.streams import Stream

    return Stream(list(prompt), 4, None, draft=True)


def test_the_concurrent_decoder_resumes_from_a_longer_disk_state(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import disk

    d, n = store(tmp_path), 1300
    src = noise(slot())
    snap, tail = kept(n)
    disk.persist(d, [(list(range(n)), src, snap, tail)], lambda: False)
    m, resumed = decoder(monkeypatch, d, [slot(), slot()])
    m.kept = [(list(range(400)), m.free.pop(), *kept(400, seed=1))]     # memory holds a shorter prefix
    s = stream([*range(n), 9, 9])
    m.admit(s)
    assert s.cached == n and s.disk_tokens == n and s.stats()["disk_tokens"] == n
    assert same_resume(resumed[0], snap, tail)
    assert all(same(a, b) for a, b in zip(disk.rows(s.st, n, n - 1), disk.rows(src, n, n - 1)))
    assert [len(k[0]) for k in m.kept if k[1] is s.st] == [n]          # kept in memory too, for the next turn


def test_a_longer_state_in_memory_wins_over_the_disk(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import disk

    d = store(tmp_path)
    disk.persist(d, [(list(range(1100)), noise(slot()), *kept(1100))], lambda: False)
    m, resumed = decoder(monkeypatch, d, [slot()])
    held = slot()
    snap, tail = kept(1200, seed=3)
    m.kept = [(list(range(1200)), held, snap, tail)]
    s = stream([*range(1300)])
    m.admit(s)
    assert s.st is held and s.cached == 1200 and s.disk_tokens == 0
    assert resumed[0]["state"] is snap


def test_a_disk_state_that_fails_prefills_the_whole_prompt(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import disk

    d, n = store(tmp_path), 1100
    disk.persist(d, [(list(range(n)), noise(slot()), *kept(n))], lambda: False)
    name = d.entries[0].meta["views"][0]["pieces"][0]
    (d.pieces / name).unlink()
    m, resumed = decoder(monkeypatch, d, [slot()])
    s = stream([*range(n + 5)])
    m.admit(s)
    assert s.cached == 0 and resumed == [None] and m.kept == []


def test_a_disk_that_cannot_record_a_use_still_resumes(tmp_path, monkeypatch, capsys):
    """ENOSPC or EROFS on the JSON rewrite after a read: the state resumes, a line says so once, no slot leaks."""

    from pathlib import Path

    from tensorfold.families.qwen4_exp.cuda import disk

    d, n = store(tmp_path), 1300
    disk.persist(d, [(list(range(n)), noise(slot()), *kept(n))], lambda: False)
    real = Path.write_text

    def full(self, *args, **kw):
        if self.parent == d.states:
            raise OSError(28, "No space left on device")
        return real(self, *args, **kw)

    monkeypatch.setattr(Path, "write_text", full)
    for prompt in ([*range(n), 9], [*range(n), 8]):                  # two decoders, as after two restarts
        m, resumed = decoder(monkeypatch, d, [slot(), slot()])
        s = stream(prompt)
        m.admit(s)
        assert s.cached == s.disk_tokens == n and resumed[0] is not None
        assert len(m.free) == 1 and all(f is not s.st for f in m.free)
    assert capsys.readouterr().out.count("last use not recorded") == 1
    assert len(d.entries) == 1 and not list(d.states.glob("*.tmp"))


@pytest.mark.parametrize("error", ["os", "oom"])
def test_a_read_the_disk_or_the_gpu_fails_prefills_and_keeps_the_state(tmp_path, monkeypatch, capsys, error):
    import torch

    from tensorfold.families.qwen4_exp.cuda import disk

    d, n = store(tmp_path), 1100
    disk.persist(d, [(list(range(n)), noise(slot()), *kept(n))], lambda: False)

    def fail(name, into):
        raise OSError(5, "Input/output error") if error == "os" else torch.cuda.OutOfMemoryError("out of memory")

    monkeypatch.setattr(d, "_read", fail)
    m, resumed = decoder(monkeypatch, d, [slot(), slot()])
    s = stream([*range(n + 5)])
    m.admit(s)
    assert s.cached == 0 and resumed == [None] and m.kept == []
    assert s in m.filling and len(m.free) == 1                       # the request holds one slot, as a prefill does
    assert "did not read" in capsys.readouterr().out
    assert len(d.entries) == 1                                         # the state itself is sound: it stays


def test_a_disk_read_that_raises_returns_the_slot(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import disk

    d, n = store(tmp_path), 1100
    disk.persist(d, [(list(range(n)), noise(slot()), *kept(n))], lambda: False)
    m, _ = decoder(monkeypatch, d, [slot(), slot()])

    def broken(entry, st):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    m._load_disk = broken
    with pytest.raises(RuntimeError, match="illegal"):
        m.admit(stream([*range(n + 5)]))
    assert len(m.free) == 2 and m.filling == [] and m.kept == []


def test_the_concurrent_decoder_writes_only_with_no_stream_live(tmp_path, monkeypatch):
    d = store(tmp_path)
    m, _ = decoder(monkeypatch, d, [])
    st = noise(slot())
    m.kept = [(list(range(1100)), st, *kept(1100))]
    m.filling = [SimpleNamespace(st=slot())]
    assert m.persist(lambda: False) == 0
    m.filling = []
    assert m.persist(lambda: False) == 1 and len(d.entries) == 1


# -- the serial engine -----------------------------------------------------------------------------------
def engine(d, st):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    e = object.__new__(FlashNextEngine)
    e.tp, e.vision, e.disk, e.cache, e.scheduler = 1, None, d, [], None
    e.e = SimpleNamespace(st=st)
    calls = []

    def decode(prompt, max_tokens, sampling, on_tokens, hit, *args, **kw):
        calls.append(hit)
        return {"cached": len(hit[0]) if hit else 0}

    e._decode = decode
    return e, calls


def test_the_serial_engine_writes_its_kept_states_and_reads_one_back(tmp_path):
    n = 1200
    live = noise(slot())
    snap, tail = kept(n)
    d = store(tmp_path)
    e, _ = engine(d, live)
    e.cache = [(list(range(300)), {"state": kept(300)[0], "tail": None}), (list(range(n)), {"state": snap, "tail": tail})]
    assert e.persist(lambda: False) == 1                      # the 300-token state prefills faster than it reads

    other, calls = engine(store(tmp_path), slot())            # a restart: nothing in memory
    other.cache = [(list(range(200)), {"state": kept(200)[0], "tail": None})]
    stats = other._generate_one([*range(n), 4, 4], 8, None, None, True, None, True, None, None)
    from tensorfold.families.qwen4_exp.cuda.disk import rows

    assert calls[0][0] == list(range(n)) and same_resume(calls[0][1], snap, tail)
    assert stats["disk_tokens"] == n and len(other.cache) == 1 and other.cache[0] is calls[0]
    assert all(same(a, b) for a, b in zip(rows(other.e.st, n, n - 1), rows(live, n, n - 1)))


def test_the_serial_engine_without_disk_states_writes_nothing():
    e, _ = engine(None, slot())
    e.cache = [(list(range(1200)), {"state": kept(1200)[0], "tail": None})]
    assert e.persist(lambda: False) == 0


# -- the scheduler's idle work ---------------------------------------------------------------------------
class Decoder:
    """One round answers every admitted stream with one token."""

    def __init__(self):
        self.streams = []

    def live(self):
        return len(self.streams)

    def admit(self, s):
        s.started = time.perf_counter()
        self.streams.append(s)

    def round(self):
        done, self.streams = self.streams, []
        for s in done:
            s.take([7])
        return done

    def finish(self, done):
        pass


def test_idle_work_runs_after_a_request_and_stops_for_the_next():
    from tensorfold.cuda.scheduler import Scheduler

    ran, stopped = threading.Event(), []

    def idle(stop):
        ran.set()
        deadline = time.monotonic() + 10
        while not stop() and time.monotonic() < deadline:
            time.sleep(0.005)
        stopped.append(stop())

    sched = Scheduler(Decoder(), max_streams=2, idle=idle)
    try:
        assert not ran.wait(0.4)                                  # nothing ran, nothing to write
        sched.submit([1, 2], 1, None, True, lambda new: False)
        assert ran.wait(5)
        sched.submit([1, 2], 1, None, True, lambda new: False)    # served once the work stops for it
        assert stopped == [True]
    finally:
        sched.close()


def test_a_call_runs_on_the_worker_and_raises_there_errors_here():
    from tensorfold.cuda.scheduler import Scheduler

    sched = Scheduler(Decoder(), max_streams=2)
    try:
        assert sched.call(threading.current_thread) is sched.thread
        with pytest.raises(ValueError, match="bad"):
            sched.call(lambda: (_ for _ in ()).throw(ValueError("bad")))
        sched.submit([1], 1, None, True, lambda new: False)       # the worker still serves
    finally:
        sched.close()
