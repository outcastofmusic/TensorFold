"""Nemotron 3.5 Bolt's window graphs on CPU: which window replays, which captures, which sets are kept, what the
graphs cost in admission, and that a capturing window does not price its row count."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.torch


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from tests.bolt_tiny import write

    return write(tmp_path_factory.mktemp("bolt"))


class Graph:
    def __init__(self, rows: int) -> None:
        self.rows, self.replays = rows, 0

    def replay(self) -> None:
        self.replays += 1


def _engine():
    """An Engine without a GPU: the graph bookkeeping of ``_window`` over a forward that only names its taps."""

    from tensorfold.families.nemotron_bolt.cuda.engine import Engine
    from tensorfold.families.nemotron_bolt.cuda.graphs import WindowGraphs

    e = Engine.__new__(Engine)
    e.graphs, e.mode, e.masked, e.captured, e._taps = WindowGraphs(sets=2), ("greedy",), False, False, []
    e.eager = []
    e.capturing = False

    def forward(rows: int) -> None:
        e._taps = [("capture" if e.capturing else "eager", rows)]
        if not e.capturing:
            e.eager.append(rows)

    def capture(rows: int) -> Graph:
        e.capturing = True
        forward(rows)
        e.capturing = False
        return Graph(rows)

    e._forward, e._capture = forward, capture
    return e


def test_a_window_runs_eagerly_and_captures_once_then_replays():
    e = _engine()
    e._window(3)
    assert e.eager == [3] and e.captured and len(e.graphs) == 1
    assert e._taps == [("eager", 3)]                       # the eager window's taps, not the capture's
    e._window(3)
    hit = e.graphs.get((("greedy",), False), 3)
    assert e.eager == [3] and not e.captured and hit.graph.replays == 1
    assert e._taps == [("capture", 3)]                     # the replay wrote the captured tap tensors
    e._window(4)
    assert e.eager == [3, 4] and e.captured


def test_the_sampling_mode_and_the_mask_pick_a_graph():
    e = _engine()
    e._window(2)
    e.masked = True
    e._window(2)
    e.mode = ("keyed", 20, True, False)
    e._window(2)
    assert e.eager == [2, 2, 2] and e.graphs.captures == 3


def test_graphs_off_runs_every_window_eagerly():
    e = _engine()
    e.graphs = None
    e._window(1)
    e._window(1)
    assert e.eager == [1, 1] and not e.captured


def test_the_least_recently_used_set_goes_first():
    from tensorfold.families.nemotron_bolt.cuda.graphs import Captured, WindowGraphs

    g = WindowGraphs(sets=2)
    a, b, c = ("greedy", False), ("greedy", True), (("keyed", 20, True, False), False)
    g.add(a, 1, Captured(Graph(1)))
    g.add(a, 8, Captured(Graph(8)))
    g.add(b, 1, Captured(Graph(1)))
    assert g.get(a, 8) is not None                          # a is now the most recently used set
    g.add(c, 1, Captured(Graph(1)))
    assert list(g.by_set) == [a, c] and len(g) == 3 and g.captures == 4
    assert g.get(b, 1) is None
    with pytest.raises(ValueError):
        WindowGraphs(sets=0)


def test_admission_counts_the_graphs(tiny, tmp_path):
    from tensorfold.families.nemotron_bolt.cuda.app import need_bytes
    from tensorfold.families.nemotron_bolt.cuda.engine import ROWS
    from tensorfold.families.nemotron_bolt.cuda.graphs import EXEC_BYTES, SETS, graph_bytes
    from tensorfold.families.nemotron_bolt.cuda.weights import Config
    from tests.bolt_lilicorr_fakes import write_draft

    c = Config.read(tiny)
    draft = write_draft(tmp_path / "draft")                 # three tap layers
    assert need_bytes(tiny, c, 1024, graphs=True) - need_bytes(tiny, c, 1024) == graph_bytes(c, 1024, ROWS, 0)
    assert (need_bytes(tiny, c, 1024, drafter=draft, graphs=True) - need_bytes(tiny, c, 1024, drafter=draft)
            == graph_bytes(c, 1024, ROWS, 3))
    # each kept graph holds its window's tap rows: 1 + 2 + ... + 16 rows a set
    assert graph_bytes(c, 1024, ROWS, 3) - graph_bytes(c, 1024, ROWS, 0) == SETS * 136 * 3 * c.hidden * 2
    assert graph_bytes(c, 1024, ROWS, 0) > SETS * ROWS * EXEC_BYTES
    assert graph_bytes(c, 2048, ROWS, 0) > graph_bytes(c, 1024, ROWS, 0)       # the attention partials grow


def test_bolts_graphs_fit_a_fraction_of_a_gib():
    from tensorfold.families.nemotron_bolt.cuda.graphs import graph_bytes
    from tensorfold.families.nemotron_bolt.cuda.weights import Config
    from tests.bolt_tiny import llm_config

    raw = llm_config(vocab=131072, hidden=4096, top_k=9)
    raw.update({"num_attention_heads": 32, "head_dim": 128, "mamba_num_heads": 128, "mamba_head_dim": 64,
                "n_groups": 8, "moe_latent_size": 1024})
    for b in raw["block_configs"]:
        if b["block_type"] == "moe":
            b["moe_intermediate_size"] = 2560
    c = Config.parse({"llm_config": raw})
    need = graph_bytes(c, 16896, 16, 6)                     # the served 16,384-token context and LiLiCorr's 6 taps
    assert 0.25 * (1 << 30) < need < 0.75 * (1 << 30)


def test_a_capturing_window_is_not_measured():
    from tensorfold.families.nemotron_bolt.cuda.depth import Costs, DepthPolicy

    p = DepthPolicy(7, Costs(8))
    p.done(7, 7, None, 12.0, 0.5)                           # the window captured its graph
    assert p.costs.verify[8] is None and p.costs.block == 12.0 and p.rate > 0.8
    p.done(7, 7, 90.0, 12.0, 0.5)
    assert p.costs.verify[8] == 90.0
