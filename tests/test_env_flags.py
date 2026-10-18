import pytest

from tensorfold.cuda.markers import GRID_ENV, MIN_GAP, grid_env
from tensorfold.env_flags import choice, flag


@pytest.mark.parametrize("value", ["1", "on", "TRUE", " yes "])
def test_a_flag_is_on(monkeypatch, value):
    monkeypatch.setenv("TENSORFOLD_X", value)
    assert flag("TENSORFOLD_X")


@pytest.mark.parametrize("value", [None, "", "0", "off", "False", "no"])
def test_a_flag_is_off(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("TENSORFOLD_X", raising=False)
    else:
        monkeypatch.setenv("TENSORFOLD_X", value)
    assert not flag("TENSORFOLD_X")


def test_a_flag_with_a_typo_fails_loudly(monkeypatch):
    monkeypatch.setenv("TENSORFOLD_X", "ture")
    with pytest.raises(ValueError, match="TENSORFOLD_X"):
        flag("TENSORFOLD_X")


def test_every_tandem_flag_rejects_a_typo(monkeypatch):
    from tensorfold.engine import think_loop
    from tensorfold.families.qwen4_exp.cuda import self_check
    from tensorfold.families.qwen4_exp.cuda.staircut import Speculation
    from tensorfold.server.metrics import drafters_enabled

    for name, read in [("TENSORFOLD_THINK_LOOP", think_loop.enabled), ("TENSORFOLD_SELF_CHECK", self_check.enabled),
                       ("TENSORFOLD_DRAFTER_METRICS", drafters_enabled),
                       ("TENSORFOLD_STAIRCUT", lambda: Speculation.from_env(confidence=0.7, depth=6)),
                       ("TENSORFOLD_PLAIN_GATE", lambda: Speculation.from_env(confidence=0.7, depth=6))]:
        monkeypatch.setenv(name, "ture")
        with pytest.raises(ValueError, match=name):
            read()
        monkeypatch.delenv(name)


def test_grid_anchors_zero_is_off(monkeypatch):
    monkeypatch.setenv(GRID_ENV, "0")
    assert grid_env() == 0
    monkeypatch.setenv(GRID_ENV, str(MIN_GAP - 1))
    with pytest.raises(ValueError):
        grid_env()


def test_an_unset_flag_takes_its_default(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_X", raising=False)
    assert flag("TENSORFOLD_X", default=True)
    monkeypatch.setenv("TENSORFOLD_X", "")
    assert flag("TENSORFOLD_X", default=True)
    monkeypatch.setenv("TENSORFOLD_X", "off")
    assert not flag("TENSORFOLD_X", default=True)


@pytest.mark.parametrize("value, ignore_case, got", [(None, False, "b"), ("", False, "b"), (" a ", False, "a"),
                                                     ("c", False, "c"), (" A ", True, "a"), ("C", True, "c")])
def test_a_choice_reads_one_of_its_values_or_the_default(monkeypatch, value, ignore_case, got):
    if value is None:
        monkeypatch.delenv("TENSORFOLD_X", raising=False)
    else:
        monkeypatch.setenv("TENSORFOLD_X", value)
    assert choice("TENSORFOLD_X", ("a", "b", "c"), "b", ignore_case=ignore_case) == got


@pytest.mark.parametrize("value, ignore_case", [("A", False), ("d", False), ("d", True), ("a,b", True)])
def test_a_choice_outside_its_values_raises_naming_the_flag_and_values(monkeypatch, value, ignore_case):
    monkeypatch.setenv("TENSORFOLD_X", value)
    with pytest.raises(ValueError, match="TENSORFOLD_X: one of a, b, c"):
        choice("TENSORFOLD_X", ("a", "b", "c"), "b", ignore_case=ignore_case)
