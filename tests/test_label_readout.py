"""A label-logit decision model (PelaAI/KnowLine's layout) through JointSchema's routes and sessions."""

from __future__ import annotations

import json

import pytest

jinja2 = pytest.importorskip("jinja2")
torch = pytest.importorskip("torch")

from tensorfold.server.label_readout import ANSWER, INSTRUCTION, MARKER, LabelReadout, block, render_value

TEMPLATE = jinja2.Template("{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
                           "<|im_start|>assistant\n<think>\n\n</think>\n\n")
LABELS = {c: 1000 + i for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}


def plain(text: str) -> list[int]:
    """One id a character, and a label after ``Answer: `` as its own id, as KnowLine's tokenizer gives it."""

    if text.endswith(" ") is False and len(text) >= 2 and text[-2] == " " and text[-1] in LABELS:
        return plain(text[:-2]) + [LABELS[text[-1]]]
    return [ord(c) for c in text]


class Engine:
    """Every prompt's last row scores the label the prompt's text names in ``prefer`` highest."""

    def __init__(self, prefer: dict[str, str]):
        self.prefer, self.calls = prefer, []

    def hidden_rows(self, ids):
        raise AssertionError("label readouts always resume through hidden_rows_from")

    def hidden_rows_from(self, ids, keep_at, kept=None):
        reused = len(kept) if kept is not None and tuple(ids[:len(kept)]) == kept else 0
        self.calls.append((len(ids), keep_at, reused))
        text = "".join(chr(i) for i in ids if i < 1000)
        row = torch.zeros(len(ids), 32)
        for word, label in self.prefer.items():
            if word in text:
                row[-1, ord(label) - ord("A")] = 4.0
        return row, tuple(ids[:keep_at]), reused

    def head_rows(self, ids):
        return torch.eye(32)[ids - 1000]


def readout(prefer, read_twice=False) -> LabelReadout:
    def render(turns):
        return TEMPLATE.render(messages=turns)

    return LabelReadout(Engine(prefer), render, plain, plain, read_twice)


def test_the_prompt_is_the_release_layout():
    r = readout({})
    text, keys = r._prompt({"invoice": {"total": 12}}, {"type": "noul", "instructions": "Is it paid?"}, r.labels, False)
    assert keys == ["true", "false"]
    assert text == ("<|im_start|>user\ninvoice:\n  total: 12<|im_end|>\n<|im_start|>user\n" + INSTRUCTION
                    + "\n\nQuestion: Is it paid?\nOptions:\nA. Yes: yes\nB. No: no<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n\n</think>\n" + ANSWER)      # jinja drops the template's last newline
    assert r.labels[:3] == ["A", "B", "C"] and r.label_ids[:3] == [1000, 1001, 1002]


def test_choices_keep_the_request_order_and_scores_number_their_levels():
    assert block({"type": "choice", "criteria": {"red": None, "blue": "the sky"}}, ["A", "B"]) == (
        "Question: Answer using the options below.\nOptions:\nA. red\nB. blue: the sky", ["red", "blue"])
    assert block({"type": "score", "criteria": ["low", "high"]}, ["A", "B"])[1] == ["0", "1"]
    with pytest.raises(ValueError, match="at most 1 options"):
        block({"type": "choice", "criteria": {"a": None, "b": None}}, ["A"])


def test_systemone_answers_each_question_from_its_label_logits():
    r = readout({"Is it done": "A", "Which colour": "B"})
    body = {"model": "knowline", "state": "the work so far", "questions": {
        "done": {"type": "noul", "instructions": "Is it done?"},
        "colour": {"type": "choice", "instructions": "Which colour?", "criteria": {"red": None, "blue": None}}}}
    answers = r.systemone(body)["answers"]
    assert answers["done"]["noul"] > 0.9
    assert answers["colour"]["choice"] == "blue" and answers["colour"]["confidence"] > 0.9


def test_decisions_route_answers_yes_no_questions_too():
    r = readout({"Is it done": "B"})
    body = {"input": "state", "questions": [{"id": "q", "type": "yes_no", "question": "Is it done?"}]}
    probabilities = r.decisions(body)["answers"]["q"]["probabilities"]
    assert probabilities["no"] > 0.9 and set(probabilities) == {"yes", "no"}


def test_a_session_resumes_every_question_after_the_state():
    r = readout({})
    questions = {"q": {"type": "noul", "instructions": "Done?"}}
    first = r.systemone({"model": "m", "state": "step one", "questions": questions, "session": True})
    second = r.systemone({"model": "m", "state": "step one\nstep two", "questions": questions,
                          "session": first["session"]})
    assert second["session"] == first["session"] and second["usage"]["cached_tokens"] > 0
    keep_at = r.engine.calls[-1][1]
    state_end = len(plain("<|im_start|>user\nstep one\nstep two<|im_end|>"))
    assert keep_at == state_end - 16


def test_the_resume_point_is_the_state_turn_s_end_even_when_the_state_spells_one():
    r = readout({})
    r.systemone({"model": "m", "state": "x" * 40 + "<|im_end|>" + "y" * 40,
                 "questions": {"q": {"type": "noul", "instructions": "Done?"}}, "session": True})
    keep_at = r.engine.calls[-1][1]
    assert keep_at == len(plain("<|im_start|>user\n" + "x" * 40 + "<|im_end|>" + "y" * 40 + "<|im_end|>")) - 16


def test_read_twice_writes_short_states_twice():
    r = readout({}, read_twice=True)
    r.systemone({"model": "m", "state": "short", "questions": {"q": {"type": "noul", "instructions": "Done?"}}})
    once = len(r._prompt("short", {"type": "noul", "instructions": "Done?"}, r.labels, False)[0])
    assert r.engine.calls[-1][0] > 2 * once - 200


def test_the_marker_names_its_layout(tmp_path):
    (tmp_path / MARKER).write_text(json.dumps({"layout": "other"}))
    with pytest.raises(ValueError, match="unknown layout"):
        LabelReadout.load(tmp_path, Engine({}), lambda turns: "", plain, plain)


def test_objects_render_as_indented_text():
    assert render_value({"a": {"b": [1, "two"]}, "c": "d"}) == "a:\n  b:\n    - 1\n    - two\nc: d"
