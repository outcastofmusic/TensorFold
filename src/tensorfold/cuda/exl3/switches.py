"""The EXL3 kernels' switches, shared by every family on these kernels (Flash Next EXL3, GLM-5.3).

TF_EXL3_EXPERT_DECODE (unset: off) runs the routed experts of a decode window of at most 16 rows on ``decode_kernel``,
which keeps ``grouped_kernel``'s sums bit for bit (``experts_grouped.cuh``). TF_EXL3_PDL (unset: off) launches the EXL3
kernels with programmatic dependent launch on GB10 (``pdl.cuh``), which changes no arithmetic. TF_EXL3_EXPERT_ITEMS
(unset: on) launches ``grouped_kernel``'s prompt windows over the member groups
``group_kernel`` lists, where it launched every group a window's rows could fill: a 1,024-row prompt window spans 32
groups an expert, and a routed expert fills one of them. The programs and their sums are the same. ``0`` restores the
old launches, and any value outside env_flags' set raises. Served on Qwen EXL3 (GPU 19 Oct), EXPERT_DECODE and PDL
were 0.99-1.00x on decode, so they stay off; the item list took cold prefill 1.41x.

Three prompt-window switches (unset: on; served 8k prefill alone 1.142x, 1.041x, 1.036x, 19 Oct), each the same bytes as without it:
TF_EXL3_GROUP_SCAN groups the picks with ``group_scan_kernel`` (a count, a scan and a stable scatter on 8 blocks),
which writes ``group_kernel``'s ids, members and item list: on Flash Next's 1,024-row windows ``group_kernel`` held
513 experts on 129 threads of one block and took 2.3 ms a window, 901 ms of an 8,192-token prefill's 6,887 (nsys,
19 Oct). TF_EXL3_DOWN_FUSE runs the down epilogue inside the down projection's programs (no fp32 partials written and
read again, no separate launch: 315 ms of the 6,887). TF_EXL3_GATEUP_FUSE has the last of an (item, n block)'s gate
and up programs add their splits and run the gate/up epilogue (433 ms of the 6,887). Both fusions take the listed
layout (TF_EXL3_EXPERT_ITEMS) at the prompt setting.

TF_EXL3_GROUP_SCAN_DECODE (unset: on; served four streams 1.026x alone, 20 Oct) groups decode windows (up to PROMPT_ROWS rows: verify windows, four streams'
rounds) on ``group_scan_kernel`` too, which TF_EXL3_GROUP_SCAN takes on prompt windows only. ``group_kernel`` costs
0.22 us a pick: 18 us a layer at 7 rows (0.87 ms of a 51.6 ms drafted round, tf-runs/profile exl3-mtp) and about
68 us at four streams' 28 rows. The same integers either way.
TF_EXL3_LINEAR_PAIR (unset: on; served four streams 1.019x alone, 20 Oct) runs the EXL3 linear's windows of more than 16 rows in passes of 32 rows, decoding
each k step's weights once for both 16-row tiles, where each 16-row pass read and decoded every weight again. Each
row keeps its mma chain, warp order and split order, so its bytes.

An engine reads every switch once at startup (``read``) and installs them (``use``). The kernels' Python callers consult
``current()`` at each launch, so a CUDA graph keeps the setting it was captured with. A test flips a setting on one
load by calling ``use`` and building a new engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from tensorfold.env_flags import flag

EXPERT_DECODE_ENV = "TF_EXL3_EXPERT_DECODE"
PDL_ENV = "TF_EXL3_PDL"
EXPERT_ITEMS_ENV = "TF_EXL3_EXPERT_ITEMS"
GROUP_SCAN_ENV = "TF_EXL3_GROUP_SCAN"
DOWN_FUSE_ENV = "TF_EXL3_DOWN_FUSE"
GATEUP_FUSE_ENV = "TF_EXL3_GATEUP_FUSE"
GROUP_SCAN_DECODE_ENV = "TF_EXL3_GROUP_SCAN_DECODE"
LINEAR_PAIR_ENV = "TF_EXL3_LINEAR_PAIR"


@dataclass(frozen=True)
class Switches:
    expert_decode: bool = False
    pdl: bool = False
    expert_items: bool = True
    group_scan: bool = False
    down_fuse: bool = False
    gateup_fuse: bool = False
    group_scan_decode: bool = False
    linear_pair: bool = False


_current = Switches()


def read() -> Switches:
    """Every switch from the environment; a typo raises before any weight is read."""

    return Switches(expert_decode=flag(EXPERT_DECODE_ENV, False), pdl=flag(PDL_ENV, False),
                    expert_items=flag(EXPERT_ITEMS_ENV, True), group_scan=flag(GROUP_SCAN_ENV, True),
                    down_fuse=flag(DOWN_FUSE_ENV, True), gateup_fuse=flag(GATEUP_FUSE_ENV, True),
                    group_scan_decode=flag(GROUP_SCAN_DECODE_ENV, True),
                    linear_pair=flag(LINEAR_PAIR_ENV, True))


def use(switches: Switches) -> Switches:
    """Install ``switches`` for the launches that follow; returns the previous setting."""

    global _current
    previous, _current = _current, switches
    return previous


def current() -> Switches:
    return _current
