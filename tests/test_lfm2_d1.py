"""The LFM2 (d1) decision family on the CPU: Liquid AI's prompt and readout, the tree forward against transformers'
Lfm2 on each question's own prompt, the engine's answers, and the HTTP routes (System One, decisions, refusals)."""

from __future__ import annotations

import http.client
import json
import math
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tokenizers")
pytest.importorskip("transformers")

from tensorfold import families
from tensorfold.families import lfm2
from tensorfold.families.lfm2.decisions import Format, answer, question
from tensorfold.families.lfm2.engine import Engine
from tensorfold.families.lfm2.model import Config, Model, Tree
from tensorfold.server.decisions import DecisionError
from tests import lfm2_tiny

pytestmark = pytest.mark.torch

NOUL = {"type": "noul", "instructions": "Is the customer asking for a refund?"}
TEAM = {"type": "choice", "instructions": "Which team should handle this?",
        "criteria": {"billing": "Charges, refunds, invoices", "technical": "App or site faults", "fraud": None}}
URGENCY = {"type": "score", "instructions": "How urgent is this?",
           "criteria": ["Can wait", "Today", "Blocking the customer now"]}
STATE = "I was charged twice this month, please refund one of them."


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return lfm2_tiny.write(tmp_path_factory.mktemp("d1"))


@pytest.fixture(scope="module")
def engine(tiny):
    return Engine(tiny, device="cpu")


@pytest.fixture(scope="module")
def fmt():
    return Format(lfm2_tiny.tokenizer())


# ---------------------------------------------------------------- the prompt (Liquid AI's prompt.py, its defaults)

def test_prompt_pieces_are_liquids(fmt):
    assert fmt.prefix(STATE) == f"<|startoftext|><|im_start|>user\n{STATE}\n\n\nQUESTION:\n"
    assert fmt.prefix({"ticket": 7}) == '<|startoftext|><|im_start|>user\n{\n  "ticket": 7\n}\n\n\nQUESTION:\n'
    assert fmt.prefix(None, "<image>") == "<|startoftext|><|im_start|>user\n<image>"
    assert fmt.suffix(question(NOUL)) == ("Is the customer asking for a refund?\n\nReply with yes or no only."
                                          "<|im_end|>\n<|im_start|>assistant\n")
    assert fmt.question_text(question(TEAM)) == (
        "Which team should handle this?\n\nOptions:\nA Charges, refunds, invoices\nB App or site faults\nC fraud\n\n"
        "Reply with the option code only.")
    assert fmt.question_text(question(URGENCY)) == (
        "How urgent is this?\n\n0 Can wait\n1 Today\n2 Blocking the customer now\n\n"
        "Reply with a single digit 0-2 only.")
    sided = {**NOUL, "criteria": {"true": "a refund is asked for"}}
    assert fmt.question_text(question(sided)).startswith(
        "Is the customer asking for a refund?\nYes: a refund is asked for\nNo: None\n\n")
    assert "Yes:" not in fmt.question_text(question({**NOUL, "criteria": {}}))


def test_option_codes_are_distinct_single_tokens(fmt):
    assert [code for code, _ in fmt.codes(("x", "y", "z"))] == ["x", "y", "z"]     # single letters stay
    assert [code for code, _ in fmt.codes(("one", "two"))] == ["A", "B"]
    many = fmt.codes(tuple(f"opt{i}" for i in range(30)))      # "00".. are two tokens here: the pool's next free
    ids = [tid for _, tid in many]
    assert len(set(ids)) == 30 and all(len(fmt.encode(code)) == 1 for code, _ in many)


def test_readout_reads_each_options_forms(fmt):
    yes, no = fmt.readout(question(NOUL))
    assert {fmt.tok.decode([t]) for t in yes} == {"yes", "Yes", "YES"}
    assert {fmt.tok.decode([t]) for t in no} == {"no", "No", "NO"}
    groups = fmt.readout(question(TEAM))
    assert [fmt.tok.decode(g) for g in groups] == ["A A", "B B", "C C"]   # "A" and " A"
    assert [fmt.tok.decode(g) for g in fmt.readout(question(URGENCY))] == ["0", "1", "2"]


@pytest.mark.parametrize("raw,message", [
    ("refund?", "is an object"),
    ({"type": "noul"}, "needs instructions"),
    ({"type": "maybe", "instructions": "x"}, "unknown question type"),
    ({"type": "score", "instructions": "x", "criteria": ["only"]}, "2 to 10 levels"),
    ({"type": "score", "instructions": "x", "criteria": [str(i) for i in range(11)]}, "2 to 10 levels"),
    ({"type": "choice", "instructions": "x", "criteria": {}}, "non-empty object"),
    ({"type": "noul", "instructions": "x", "criteria": ["yes"]}, "true and false"),
])
def test_malformed_questions_are_refused(raw, message):
    with pytest.raises(DecisionError, match=message):
        question(raw)


def test_answers_are_a_softmax_over_the_options():
    assert answer(question(NOUL), [0.0, 0.0]) == {"type": "noul", "noul": 0.5}
    chosen = answer(question(TEAM), [1.0, 3.0, 2.0])
    assert chosen["choice"] == "technical" and math.isclose(sum(chosen["probabilities"].values()), 1.0)
    scored = answer(question(URGENCY), [0.0, 0.0, math.log(2.0)])
    assert math.isclose(scored["score"], 0.25 * 1 + 0.5 * 2) and scored["legend"]["2"] == "Blocking the customer now"


# ---------------------------------------------------------------- the forward

def test_tree_tables():
    tree = Tree(3, [1, 3], "cpu", taps=3)
    assert tree.positions.tolist() == [0, 1, 2, 3, 3, 4, 5]
    assert tree.leaves.tolist() == [3, 6]
    # predecessors, oldest first: the second branch's first token continues the trunk, not the first branch
    assert tree.prev.tolist() == [[-1, -1], [-1, 0], [0, 1], [1, 2], [1, 2], [2, 4], [4, 5]]
    assert tree.mask.tolist() == [[True, True, True, True, False, False, False],
                                  [True, True, True, False, True, False, False],
                                  [True, True, True, False, True, True, False],
                                  [True, True, True, False, True, True, True]]


@pytest.mark.parametrize("change,eps,thetas", [
    ({}, 1e-5, (1e6, 1e4)),
    ({"norm_eps": None, "block_norm_eps": 1e-6}, 1e-6, (1e6, 1e4)),
    ({"norm_eps": None}, 1e-5, (1e6, 1e4)),
    ({"rope_parameters": None, "rope_theta": 5e5}, 1e-5, (5e5, 5e5)),
    ({"rope_parameters": None}, 1e-5, (1e6, 1e6)),
    ({"rope_parameters": {"rope_type": "default", "rope_theta": 1e7}}, 1e-5, (1e7, 1e7)),
])
def test_both_families_read_the_core_text_fields_alike(change, eps, thetas):
    """d1's Config and LFM2-MoE's read hidden size, heads, layers, taps, norm epsilon and RoPE theta from one parser,
    in one order: norm_eps, else block_norm_eps, else 1e-5; rope_parameters' theta, else
    rope_theta, else 1e6."""

    from tensorfold.families.lfm2_moe.config import Config as MoeConfig
    from tests import lfm2_moe_tiny

    text = {**lfm2_tiny.text_config(300), **change}
    d1 = Config.of(text)
    assert (d1.eps, d1.theta, d1.taps, d1.head_dim, d1.layers) == (eps, thetas[0], 3, 16, lfm2_tiny.LAYERS)
    moe = MoeConfig.parse({**lfm2_moe_tiny.config(), **change})
    assert (moe.eps, moe.theta, moe.conv_taps, moe.head_dim) == (eps, thetas[1], 3, 64)


def _reference(tiny):
    """transformers' Lfm2ForCausalLM in float32 with the tiny checkpoint's language weights."""

    from safetensors.torch import load_file
    from transformers import Lfm2Config, Lfm2ForCausalLM

    text = json.loads((tiny / "config.json").read_text())["text_config"]
    hf = Lfm2ForCausalLM(Lfm2Config(**text)).float().eval()
    weights = {k.replace("model.language_model.", "model."): v.float()
               for k, v in load_file(str(tiny / "model.safetensors")).items() if k.startswith("model.language_model.")}
    missing = hf.load_state_dict(weights, strict=False).missing_keys
    assert missing in ([], ["lm_head.weight"])
    hf.tie_weights()
    ours = Model.from_tensors(Config.of(text), {k: v for k, v in weights.items()}, "cpu", prefix="model.")
    return hf, ours


@pytest.mark.parametrize("trunk,lengths", [(7, [1, 4, 9]), (0, [5, 2]), (1, [3]), (2, [1, 1, 1])])
def test_tree_forward_equals_each_question_alone_in_transformers(tiny, trunk, lengths):
    hf, ours = _reference(tiny)
    g = torch.Generator().manual_seed(trunk * 31 + len(lengths))
    vocab = ours.cfg.vocab
    shared = torch.randint(0, vocab, (trunk,), generator=g).tolist()
    branches = [torch.randint(0, vocab, (n,), generator=g).tolist() for n in lengths]
    x = ours.embed_ids(shared + [t for b in branches for t in b])
    got = ours.forward(x, Tree(trunk, lengths, "cpu"))
    for row, branch in zip(got, branches):
        with torch.no_grad():
            want = hf.model(input_ids=torch.tensor([shared + branch])).last_hidden_state[0, -1]
            logits = hf(input_ids=torch.tensor([shared + branch])).logits[0, -1]
        assert torch.allclose(row, want, atol=1e-4, rtol=1e-4), (row - want).abs().max()
        assert torch.allclose(ours.logits(row[None])[0], logits, atol=1e-3, rtol=1e-4)


def test_long_branch_sets_take_blocks_of_rows(tiny, monkeypatch):
    from tensorfold.families.lfm2 import model as module

    _, ours = _reference(tiny)
    x = ours.embed_ids(list(range(20)))
    whole = ours.forward(x, Tree(4, [5, 6, 5], "cpu"))
    monkeypatch.setattr(module, "ROWS", 3)
    assert torch.allclose(ours.forward(x, Tree(4, [5, 6, 5], "cpu")), whole, atol=1e-6)


# ---------------------------------------------------------------- the engine

def test_engine_answers_three_questions_in_one_forward(engine):
    out = engine.decide(STATE, {"refund": NOUL, "team": TEAM, "urgency": URGENCY})
    answers = out["answers"]
    assert [answers[k]["type"] for k in ("refund", "team", "urgency")] == ["noul", "choice", "score"]
    assert 0 < answers["refund"]["noul"] < 1
    assert math.isclose(sum(answers["team"]["probabilities"].values()), 1.0)
    assert answers["team"]["choice"] in TEAM["criteria"] and 0 <= answers["urgency"]["score"] <= 2
    f = engine.fmt
    read = len(f.encode(f.prefix(STATE))) + sum(len(f.encode(f.suffix(question(q)))) for q in (NOUL, TEAM, URGENCY))
    assert out["usage"] == {"input_tokens": read, "output_tokens": 0}


def test_a_request_s_questions_read_the_head_once(tiny, monkeypatch):
    """Every question's options from one head matmul over the union of their tokens (TF_D1_HEAD=union)."""

    engine = _head_engine(tiny, monkeypatch, "union")
    calls: list[tuple[int, int]] = []
    own = engine.model.logits

    def counted(hidden, ids=None):
        calls.append((hidden.shape[0], len(ids) if ids is not None else -1))
        return own(hidden, ids)

    monkeypatch.setattr(engine.model, "logits", counted)
    engine.decide(STATE, {"refund": NOUL, "team": TEAM, "urgency": URGENCY})
    f = engine.fmt
    union = {t for q in (NOUL, TEAM, URGENCY) for forms in f.readout(question(q)) for t in forms}
    assert calls == [(3, len(union))]


def test_a_question_reads_the_same_alone_and_beside_others(engine, tiny):
    together = engine.decide(STATE, {"refund": NOUL, "team": TEAM, "urgency": URGENCY})["answers"]
    alone = engine.decide(STATE, {"team": TEAM})["answers"]["team"]
    for name, p in alone["probabilities"].items():
        assert abs(p - together["team"]["probabilities"][name]) < 2e-2
    f = engine.fmt
    longest = max(len(f.encode(f.suffix(question(q)))) for q in (NOUL, TEAM, URGENCY))
    split = Engine(tiny, device="cpu", token_budget=len(f.encode(f.prefix(STATE))) + longest)    # a question a forward
    apart = split.decide(STATE, {"refund": NOUL, "team": TEAM, "urgency": URGENCY})["answers"]
    assert apart["team"]["choice"] == together["team"]["choice"]
    assert abs(apart["refund"]["noul"] - together["refund"]["noul"]) < 2e-2


def _head_engine(tiny, monkeypatch, head):
    monkeypatch.setenv("TF_D1_HEAD", head)
    return Engine(tiny, device="cpu")


def test_head_choice_defaults_to_per_question_and_refuses_unknown_values(tiny, monkeypatch):
    monkeypatch.delenv("TF_D1_HEAD", raising=False)
    assert Engine(tiny, device="cpu").head == "per_question"
    with pytest.raises(ValueError, match="TF_D1_HEAD='fused': one of union, per_question, per_question_sync"):
        _head_engine(tiny, monkeypatch, "fused")


@pytest.mark.parametrize("head,reads", [("union", 1), ("per_question", 1), ("per_question_sync", 3)])
def test_each_head_choice_reads_back_as_named(tiny, monkeypatch, head, reads):
    engine = _head_engine(tiny, monkeypatch, head)
    f = engine.fmt
    groups = [f.readout(question(q)) for q in (NOUL, TEAM, URGENCY)]
    hidden = engine._rows(f.encode(f.prefix(STATE)), [f.encode(f.suffix(question(q))) for q in (NOUL, TEAM, URGENCY)])
    calls = []
    own = torch.Tensor.tolist

    def counted(self):
        calls.append(tuple(self.shape))
        return own(self)

    monkeypatch.setattr(torch.Tensor, "tolist", counted)
    scores = engine._scores(hidden, groups)
    assert len(calls) == reads and [len(s) for s in scores] == [len(g) for g in groups]


def test_per_question_gives_per_question_sync_s_bits(tiny, monkeypatch):
    f = Format(lfm2_tiny.tokenizer())
    questions = [NOUL, TEAM, URGENCY, {"type": "choice", "instructions": "Which colour?",
                                       "criteria": {"red": None, "green": None, "blue": None, "grey": None}}]
    groups = [f.readout(question(q)) for q in questions]
    got = {}
    for head in ("union", "per_question", "per_question_sync"):
        engine = _head_engine(tiny, monkeypatch, head)
        hidden = engine._rows(f.encode(f.prefix(STATE)), [f.encode(f.suffix(question(q))) for q in questions])
        got[head] = engine._scores(hidden, groups)
    assert got["per_question"] == got["per_question_sync"]          # float equality, every logit
    for a, b in zip(got["union"], got["per_question_sync"]):
        assert all(math.isclose(x, y, abs_tol=5e-2) for x, y in zip(a, b))


def test_engine_answers_match_transformers_on_the_full_prompt(engine, tiny):
    hf, _ = _reference(tiny)
    f = engine.fmt
    q = question(TEAM)
    ids = f.encode(f.prefix(STATE) + f.suffix(q))
    with torch.no_grad():
        logits = hf(input_ids=torch.tensor([ids])).logits[0, -1]
    want = answer(q, [max(float(logits[t]) for t in forms) for forms in f.readout(q)])
    got = engine.decide(STATE, {"team": TEAM})["answers"]["team"]
    assert got["choice"] == want["choice"]
    for name in want["probabilities"]:
        assert abs(got["probabilities"][name] - want["probabilities"][name]) < 3e-2   # bf16 against float32


def test_state_past_the_context_is_refused(tiny):
    small = Engine(tiny, device="cpu", context=32)
    with pytest.raises(DecisionError, match="32-token context"):
        small.decide("word " * 40, {"refund": NOUL})
    with pytest.raises(DecisionError, match="non-empty object"):
        small.decide(STATE, {})
    with pytest.raises(DecisionError, match="no vision tower"):
        small.decide(None, {"refund": NOUL}, images=[object()])


def test_score_labels_reads_the_whole_vocabulary(engine):
    a = engine.fmt.encode("<|startoftext|><|im_start|>user\nIs it late?<|im_end|>\n<|im_start|>assistant\n")
    b = engine.fmt.encode("<|startoftext|><|im_start|>user\nIs it early?<|im_end|>\n<|im_start|>assistant\n")
    labels = engine.fmt.readout(question(NOUL))
    one = engine.score_labels(a, [labels[0][0], labels[1][0]])
    both = engine.score_labels_many([(a, [labels[0][0], labels[1][0]]), (b, [labels[0][0]])])
    assert all(math.isclose(x, y, abs_tol=2e-2) for x, y in zip(one[0], both[0][0]))
    assert one[1] > max(one[0]) and math.isclose(one[1], both[0][1], abs_tol=2e-2)
    with pytest.raises(DecisionError, match="cannot be empty"):
        engine.score_labels([], [1])


def test_engine_writes_no_text(engine):
    with pytest.raises(DecisionError, match="writes no text"):
        engine.generate([1, 2], 4, None, lambda tokens: None)


# ---------------------------------------------------------------- the family

def test_family_is_found_by_its_model_type(tiny):
    family = families.detect(tiny)
    assert family.module == "tensorfold.families.lfm2" and not family.lanes
    assert families.backends_of(family) == ("cuda",)
    lfm2.check(tiny)


@pytest.mark.parametrize("change,message", [
    ({"quantization_config": {"quant_method": "fp8"}}, "own bf16 weights"),
    ({"text_config": {"attention_bias": True}}, "an attention bias"),
    ({"text_config": {"layer_types": ["conv", "sliding_attention"]}}, "neither conv nor full_attention"),
    ({"text_config": {"num_key_value_heads": 3}}, "no grouped attention"),
    ({"text_config": {"dtype": "float16"}}, "a dtype other than bfloat16"),
])
def test_checkpoints_the_forward_does_not_read_are_refused(tiny, tmp_path, change, message):
    cfg = json.loads((tiny / "config.json").read_text())
    text = {**cfg["text_config"], **change.pop("text_config", {})}
    (tmp_path / "config.json").write_text(json.dumps({**cfg, **change, "text_config": text}))
    with pytest.raises(ValueError, match=message):
        lfm2.check(tmp_path)


@pytest.mark.parametrize("options,message", [
    ({"tp": 2}, "one GPU"), ({"drafter": "/models/x"}, "--drafter"), ({"vision": True}, "drop the flag"),
    ({"parallel": 4}, "parallel"), ({"kv_dtype": "int8"}, "kv_dtype"),
])
def test_serve_options_without_a_meaning_are_refused(tiny, options, message):
    with pytest.raises(ValueError, match=message):
        lfm2.cuda_engine(tiny, **options)


def test_token_budget_variable_is_checked(tiny, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_LFM2_TOKEN_BUDGET", "lots")
    with pytest.raises(ValueError, match="positive token count"):
        lfm2.cuda_engine(tiny)


def test_the_chat_template_renders_its_generation_block(tiny):
    from tensorfold.cuda.chat_template import ChatTemplate

    rendered = ChatTemplate(tiny).render([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"},
                                          {"role": "user", "content": "and?"}], tools=None, enable_thinking=False)
    assert rendered == ("<|startoftext|><|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\nok<|im_end|>\n"
                        "<|im_start|>user\nand?<|im_end|>\n<|im_start|>assistant\n")


# ---------------------------------------------------------------- pictures

def test_picture_features_fill_their_placeholders_as_transformers_does(tmp_path):
    from transformers import Lfm2VlConfig
    from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlModel


    folder = lfm2_tiny.write(tmp_path / "vl", vision=True)
    cfg = json.loads((folder / "config.json").read_text())
    image = cfg["image_token_id"]
    g = torch.Generator().manual_seed(3)
    pixels = torch.randn((1, 20, 3 * 16 * 16), generator=g)          # 16 real patches (4 x 4), padded to 20
    mask = torch.tensor([[True] * 16 + [False] * 4])
    shapes = torch.tensor([[4, 4]])

    class Processor:                                  # what Lfm2VlProcessor returns for one 64 x 64 picture
        def __call__(self, text, images, return_tensors, add_special_tokens):
            ids = Format(lfm2_tiny.tokenizer()).encode(text[0].replace("<image>", ""))
            return {"input_ids": torch.tensor([ids[:3] + [image] * 4 + ids[3:]]), "pixel_values": pixels,
                    "pixel_attention_mask": mask, "spatial_shapes": shapes}

        def apply_chat_template(self, messages, **_):
            return "<|im_start|>user\n<image>\x00<|im_end|>\n"

    engine = Engine(folder, device="cpu", processor=Processor())
    pictures = engine.pictures
    assert pictures.markup(1) == "<image>"
    ids, embeds = pictures.trunk(engine.model, engine.fmt.prefix(None, "<image>"), [object()])
    assert ids.count(image) == 4 and embeds.shape == (len(ids), 64)

    hf = Lfm2VlModel(Lfm2VlConfig(**{k: v for k, v in cfg.items() if k != "architectures"})).to(torch.bfloat16).eval()
    from safetensors.torch import load_file

    weights = {k.removeprefix("model."): v for k, v in load_file(str(folder / "model.safetensors")).items()
               if k.startswith(("model.vision_tower.", "model.multi_modal_projector."))}
    assert not [k for k in hf.load_state_dict(weights, strict=False).unexpected_keys]
    with torch.no_grad():
        want = hf.get_image_features(pixel_values=pixels.to(torch.bfloat16), spatial_shapes=shapes,
                                     pixel_attention_mask=mask).pooler_output[0]
    slots = torch.tensor(ids) == image
    assert torch.allclose(embeds[slots].float(), want.float(), atol=2e-2)
    assert torch.equal(embeds[~slots], engine.model.embed_ids([t for t in ids if t != image]))
    out = engine.decide(None, {"cats": {"type": "choice", "instructions": "How many cats are there?",
                                        "criteria": {"one": "One", "two": "Two"}}}, images=[_png()])
    assert out["answers"]["cats"]["choice"] in ("one", "two")


def _png():
    from PIL import Image

    return Image.new("RGB", (64, 64), (200, 30, 30))


# ---------------------------------------------------------------- HTTP

@contextmanager
def _serving(app):
    from tensorfold.cuda.http import make_handler

    class Quiet(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            pass

    httpd = Quiet(("127.0.0.1", 0), make_handler(app))
    worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield httpd.server_port
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join()


def _post(port, route, body):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        connection.request("POST", route, json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read().decode() or "{}")
    finally:
        connection.close()


@pytest.fixture(scope="module")
def app(engine, tiny):
    return lfm2.CUDA_APP(engine, tiny, "d1-3b")


def test_systemone_route_answers_named_questions(app):
    with _serving(app) as port:
        status, body = _post(port, "/v1/systemone", {"model": "d1-3b", "state": STATE,
                                                     "questions": {"refund": NOUL, "urgency": URGENCY}})
    assert status == 200 and body["model"] == "d1-3b"
    assert set(body["answers"]) == {"refund", "urgency"} and body["usage"]["output_tokens"] == 0


def test_a_chat_state_keeps_its_messages_and_gives_up_its_pictures(app, monkeypatch):
    seen = {}
    monkeypatch.setattr(app.engine, "decide", lambda state, questions, images: seen.update(
        state=state, images=images) or {"answers": {}, "usage": {}})
    url = "data:image/png;base64,iVBORw0KGgo="
    state = {"messages": [{"role": "user", "content": [{"type": "text", "text": "look"},
                                                       {"type": "image_url", "image_url": {"url": url}}]}]}
    monkeypatch.setattr("tensorfold.families.lfm2.app.load_images", lambda sources, limits: [])
    app.systemone({"state": state, "questions": {"refund": NOUL}})
    assert seen["state"] == {"messages": [{"role": "user", "content": [{"type": "text", "text": "look"}]}]}
    with pytest.raises(DecisionError, match="video_url is not supported"):
        app.systemone({"state": [{"role": "user", "content": [{"type": "video_url", "video_url": {"url": url}}]}],
                       "questions": {"refund": NOUL}})


@pytest.mark.parametrize("body,message", [
    ({"questions": {"refund": NOUL}}, "needs a state"),
    ({"state": STATE}, "needs questions"),
    ({"state": STATE, "questions": {"refund": {"type": "noul"}}}, "needs instructions"),
    ({"state": STATE, "questions": {"refund": NOUL}, "images": "data:,"}, "array of data URLs"),
    ({"state": STATE, "questions": {"refund": NOUL}, "videos": ["x"]}, "videos are not supported"),
])
def test_malformed_systemone_requests_are_400(app, body, message):
    with _serving(app) as port:
        status, reply = _post(port, "/v1/systemone", body)
    assert status == 400 and message in reply["error"]["message"]


def test_decisions_route_scores_labels(app):
    body = {"input": STATE, "questions": [{"id": "refund", "type": "yes_no", "question": "a refund is asked for"},
                                          {"id": "team", "type": "choice", "question": "Which team?",
                                           "options": [{"name": "billing"}, {"name": "technical"}]}]}
    with _serving(app) as port:
        status, reply = _post(port, "/v1/decisions", body)
    assert status == 200, reply
    assert set(reply["answers"]) == {"refund", "team"} and reply["usage"]["completion_tokens"] == 0
    assert all(0 < a["label_mass"] <= 1 for a in reply["answers"].values())


@pytest.mark.parametrize("route,body", [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/completions", {"prompt": "hi"}),
    ("/v1/messages", {"model": "d1-3b", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}),
])
def test_text_requests_are_refused(app, route, body):
    with _serving(app) as port:
        status, reply = _post(port, route, body)
    assert status == 400 and "writes no text" in json.dumps(reply)


def test_models_and_health_answer(app):
    """run.sh's readiness check reads /v1/models."""

    with _serving(app) as port:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        try:
            connection.request("GET", "/v1/models")
            models = json.loads(connection.getresponse().read())
            connection.request("GET", "/health")
            health = connection.getresponse()
            assert health.status == 200 and json.loads(health.read())
        finally:
            connection.close()
    assert [m["id"] for m in models["data"]] == ["d1-3b"]


def test_the_server_sets_no_delay_for_every_app(app):
    """A reply's headers and its small body are two writes: without TCP_NODELAY a kept connection waits out the
    client's delayed ACK (40 ms on Linux) for the body."""

    from types import SimpleNamespace

    from tensorfold.cuda.http import make_handler

    assert make_handler(app).disable_nagle_algorithm is True
    assert make_handler(SimpleNamespace()).disable_nagle_algorithm is True
