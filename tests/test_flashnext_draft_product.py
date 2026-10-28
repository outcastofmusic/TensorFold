"""The MTP chain's stop rule: a 0.7 floor per draft by default; TF_FLASHNEXT_DRAFT_PRODUCT also ends a chain before a
draft that takes the product of the drafts' probabilities below it (Mia's 0017: 0.4 over a 0.5 floor)."""

import pytest

torch = pytest.importorskip("torch")


class _Engine:
    """Drafts 10, 11, ... with the given probabilities; counts MTP steps."""

    def __init__(self, probs):
        self.probs, self.steps = probs, 0
        self.st = type("St", (), {"mtp_len": 0, "mtp_drafted": 0, "set_mtp_len": lambda self, n: setattr(self, "mtp_len", n)})()
        self.mbuf = type("Buf", (), {"streams": torch.zeros(4, 1)})()

    def sample_draft(self, logits, position, sampling):
        j = position - 100
        return 10 + j, self.probs[j]

    def mtp_forward(self, tokens, prev):
        self.steps += 1
        return None


def _chain(monkeypatch, probs, floor, product):
    from tensorfold.families.qwen4_exp.cuda import decode

    monkeypatch.setattr(decode, "absorb", lambda e, streams, next_tokens: None)
    monkeypatch.setattr(decode, "PRODUCT", product)
    return decode.draft(_Engine(probs), None, [1], 100, len(probs), None, floor)


def test_the_default_floor_ends_a_chain_before_a_draft_under_it(monkeypatch):
    assert _chain(monkeypatch, [0.9, 0.8, 0.65, 0.9], 0.7, 0.0) == [10, 11]


def test_the_product_rule_keeps_drafts_under_the_old_floor_while_the_chain_is_likely(monkeypatch):
    # products 0.9, 0.72, 0.468, 0.421, then 0.38: the fifth draft would take the chain under 0.4
    assert _chain(monkeypatch, [0.9, 0.8, 0.65, 0.9, 0.9], 0.5, 0.4) == [10, 11, 12, 13]


def test_the_floor_still_applies_under_the_product_rule(monkeypatch):
    assert _chain(monkeypatch, [0.95, 0.45, 0.99], 0.5, 0.4) == [10]


def test_the_first_draft_is_kept_even_below_both(monkeypatch):
    assert _chain(monkeypatch, [0.3, 0.9], 0.5, 0.4) == [10]
