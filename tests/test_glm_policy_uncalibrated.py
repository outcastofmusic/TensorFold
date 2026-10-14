"""GLM without calibration (--eager, TF_GLM_CALIBRATE=0): policies that resolve to a fixed one still serve, and a
request that needs the chooser's timings is refused before its prefill."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tensorfold.families.glm5_next.cuda.engine import DFLASH_POLICY, EXL3_AUTO, GlmEngine, encode_policy

pytestmark = pytest.mark.torch


def engine(*, quant="exl3", drafter=True, mtp=False):
    e = GlmEngine.__new__(GlmEngine)
    e.serial_only, e.calibrated, e.costs = False, False, None
    e.drafter = object() if drafter else None
    e.w = SimpleNamespace(cfg=SimpleNamespace(quant=quant), mtp=object() if mtp else None)
    return e


def test_auto_on_an_exl3_pack_with_a_drafter_is_the_fixed_policy_without_calibration():
    assert engine()._effective(encode_policy("auto")) == encode_policy(EXL3_AUTO)


def test_auto_without_mtp_or_timings_is_the_fixed_dflash_policy():
    assert engine(quant="affine")._effective(encode_policy("auto")) == encode_policy(DFLASH_POLICY)


def test_auto_on_an_mtp_pack_without_a_drafter_needs_no_timings():
    assert engine(quant="affine", drafter=False, mtp=True)._effective(encode_policy("auto")) == encode_policy("auto")


def test_the_chooser_without_timings_is_refused_before_the_prefill():
    with pytest.raises(ValueError, match="calibration"):
        engine(quant="affine", mtp=True)._effective(encode_policy("auto:2:8:0.03"))


def test_calibrate_off_in_the_environment_counts_as_uncalibrated(monkeypatch):
    from tensorfold.families.glm5_next.cuda import engine as module

    monkeypatch.setenv("TF_GLM_CALIBRATE", "0")
    assert module.calibrates(True, eager=False, serial_only=False) is False
    monkeypatch.delenv("TF_GLM_CALIBRATE")
    assert module.calibrates(True, eager=False, serial_only=False) is True
    assert module.calibrates(True, eager=True, serial_only=False) is False
