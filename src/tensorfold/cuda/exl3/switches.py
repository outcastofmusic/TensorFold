"""The EXL3 kernels' switches, shared by every family on these kernels (Flash Next EXL3, GLM-5.3).

TF_EXL3_EXPERT_DECODE (unset: on) runs the routed experts of a decode window of at most 16 rows on ``decode_kernel``,
which keeps ``grouped_kernel``'s sums bit for bit (``experts_grouped.cuh``). TF_EXL3_PDL (unset: on) launches the EXL3
kernels with programmatic dependent launch on GB10 (``pdl.cuh``), which changes no arithmetic. TF_EXL3_EXPERT_ITEMS
(unset: on) launches ``grouped_kernel`` (prompt windows, and decode windows past 16 rows) over the member groups
``group_kernel`` lists, where it launched every group a window's rows could fill: a 1,024-row prompt window spans 32
groups an expert, and a routed expert fills one of them. The programs and their sums are the same. ``0`` restores the
old launches, and any value outside env_flags' set raises.

An engine reads both once at startup (``read``) and installs them (``use``). The kernels' Python callers consult
``current()`` at each launch, so a CUDA graph keeps the setting it was captured with. A test flips a setting on one
load by calling ``use`` and building a new engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from tensorfold.env_flags import flag

EXPERT_DECODE_ENV = "TF_EXL3_EXPERT_DECODE"
PDL_ENV = "TF_EXL3_PDL"
EXPERT_ITEMS_ENV = "TF_EXL3_EXPERT_ITEMS"


@dataclass(frozen=True)
class Switches:
    expert_decode: bool = True
    pdl: bool = True
    expert_items: bool = True


_current = Switches()


def read() -> Switches:
    """Every switch from the environment; a typo raises before any weight is read."""

    return Switches(expert_decode=flag(EXPERT_DECODE_ENV, True), pdl=flag(PDL_ENV, True),
                    expert_items=flag(EXPERT_ITEMS_ENV, True))


def use(switches: Switches) -> Switches:
    """Install ``switches`` for the launches that follow; returns the previous setting."""

    global _current
    previous, _current = _current, switches
    return previous


def current() -> Switches:
    return _current
