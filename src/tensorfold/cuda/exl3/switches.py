"""The EXL3 kernels' one switch, shared by every family on these kernels (Flash Next EXL3, GLM-5.3).

TF_EXL3_PDL (unset: off) launches the EXL3 kernels with programmatic dependent launch on GB10 (``pdl.cuh``), which
changes no arithmetic. Served on Qwen EXL3 (GPU 19 Oct), PDL with the since-removed decode expert kernel was
0.99-1.00x on decode, so it stays off.

The routed experts' and the linear's other launch choices follow the shape alone (``experts.routed``,
``linear.row_tiles``): the switches that once turned each off were measured and removed on 25 Oct.

An engine reads the switch once at startup (``read``) and installs it (``use``). The kernels' Python callers consult
``current()`` at each launch, so a CUDA graph keeps the setting it was captured with. A test flips a setting on one
load by calling ``use`` and building a new engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from tensorfold.env_flags import flag

PDL_ENV = "TF_EXL3_PDL"


@dataclass(frozen=True)
class Switches:
    """Each field's default is its served setting, the one ``read`` takes for an unset flag."""

    pdl: bool = False


ENVS = {PDL_ENV: "pdl"}

_current = Switches()          # launches before any engine's ``use`` (tools, tests) take the served setting


def read() -> Switches:
    """The switch from the environment; a typo raises before any weight is read."""

    served = Switches()
    return Switches(**{field: flag(env, getattr(served, field)) for env, field in ENVS.items()})


def use(switches: Switches) -> Switches:
    """Install ``switches`` for the launches that follow; returns the previous setting."""

    global _current
    previous, _current = _current, switches
    return previous


def current() -> Switches:
    return _current
