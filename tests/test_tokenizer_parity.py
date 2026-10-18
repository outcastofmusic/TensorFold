"""The CUDA server's tokenizer path gives transformers' ids on an adversarial text set, decodes them back to the text,
and puts at most one BOS in a chat prompt, on every served checkpoint (CPU only).

The text set follows llama.cpp's ``CHK_TXT`` (convert_hf_to_gguf_update.py): whitespace runs, tabs, CRLF, emoji with
ZWJ sequences, CJK, Arabic, combining marks, code, digit runs, and each checkpoint's added tokens written as plain
text. The server encodes a chat prompt with ``PromptIds`` (pieces between added tokens) and a raw prompt with
``tokenizers.Tokenizer.encode``; transformers' ``AutoTokenizer`` is the reference. A missing checkpoint is skipped.
"""

from __future__ import annotations

import json
import os
import unicodedata
from functools import cache
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("jinja2")
pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from tensorfold.cuda.prompt_ids import PromptIds  # noqa: E402

MODELS = Path(os.environ.get("MODELS", str(Path.home() / "models")))
# name: (folder under $MODELS, the tokens a chat prompt may hold once: BOS or the template's own prefix)
SERVED = {
    "flashnext-exl3": ("Qwen3.8-Flash-Next-exl3-3.05bpw", ()),
    "flashnext-fp4": ("Qwen3.8-Flash-Next-FP4-Local", ()),
    "glm": ("glm53-flash-sushi-experts-mosaic-dense", ("[gMASK]", "<sop>")),
    "bolt": ("Nemotron-3.5-Bolt-74B-A9B-24GB", ()),         # its template writes no <s>, nor does transformers
    "lfm2": ("LFM2.5-24B-A2B-FP8", ("<|startoftext|>",)),
}

CHK_TXT = [
    "",
    " ",
    "  ",
    "\t",
    "\n",
    "\n\n",
    "\r\n",
    " \r\n \r\n\r\n",
    "\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n",
    "Hello world",
    " Hello world",
    "Hello World",
    "HELLO WORLD!",
    "   leading and trailing   ",
    "line one\r\nline two\r\n\r\nline four\r",
    "tab\tseparated\tvalues\t\t\tend",
    "🚀 (normal) 😶‍🌫️ (multiple emojis concatenated) ✅ 🦙🦙",
    "👨‍👩‍👧‍👦 👩🏽‍💻 🏳️‍🌈 🇳🇴🇯🇵 ❤️‍🔥 1️⃣",
    "3 33 333 3333 33333 333333 3333333 33333333 3.3 3..3 3...3 1,000,000 -0.5e-10 0x1F 1_000",
    "កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～",
    "日本語のテキストと漢字、ひらがな、カタカナ。中文：你好，世界！한국어 텍스트",
    "مرحبا بالعالم، هذا نص عربي مع أرقام ١٢٣ و 456",
    "שלום עולם",
    "нещо на Български",
    "naïve café résumé",
    "nai\u0308ve cafe\u0301 re\u0301sume\u0301",          # combining marks (NFD), not the precomposed letters
    "Z\u0351\u036b\u0343\u036a\u0302\u036b\u033d\u034f\u0334\u0319\u0324\u031e\u0349\u035a\u032f\u031e\u0320\u034dA",
    "\u00a0non-breaking\u2009thin\u200bzero-width\u2028line-sep\ufeffbom",
    "------======= '''''''```````\"\"\"\"......!!!!!!?????? ~~~~ ____ ****",
    "I've been 'told he's there, 'RE you sure? 'M not sure I'll make it, 'D you like some tea? We'Ve a'lL",
    "def f(x):\n    return {\"a\": [1, 2, 3]}  # comment\n\n\nclass A:\n\tpass\n",
    "for (int i = 0; i < n; ++i) {\n    printf(\"%d\\n\", a[i]);\n}\r\n",
    "<div class=\"x\">&amp; &lt;tag&gt;</div>\n<script>alert('x')</script>",
    "SELECT * FROM t WHERE a = 'b' AND c <> 1;",
    "https://example.com/path?q=1&r=two#frag user@example.com",
    "\\u00e9 \\n \\t \\\\ \\\"",
    "∀x∈ℝ: x² ≥ 0 ⇒ √(x²) = |x| ≠ −x ∑∫∂ ½ ⅓",
    "a" * 300,
    " " * 100 + "x" + "\n" * 50,
]


@cache
def checkpoint(name: str):
    """(folder, our tokenizer, transformers' tokenizer) for a served checkpoint, or a skip."""

    from tokenizers import Tokenizer

    folder = MODELS / SERVED[name][0]
    if not (folder / "tokenizer.json").is_file():
        pytest.skip(f"{folder} is not here")
    ours = Tokenizer.from_file(str(folder / "tokenizer.json"))     # cuda.server.App's tokenizer
    theirs = transformers.AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    return folder, ours, theirs


def added_text(ours) -> str:
    """Every added token of the checkpoint, written into one text with words and spaces around them."""

    tokens = [str(t.content) for _, t in sorted(ours.get_added_tokens_decoder().items())]
    return " x ".join(tokens) + " and <think>literal</think>  <tool_call>\n"


def texts(ours) -> list[str]:
    return [*CHK_TXT, added_text(ours)]


def normalized(ours, text: str) -> str:
    """The text a lossless tokenizer gives back: Qwen's normalizes to NFC first."""

    kind = (json.loads(ours.to_str()).get("normalizer") or {}).get("type")
    return unicodedata.normalize("NFC", text) if kind == "NFC" else text


@pytest.mark.parametrize("name", sorted(SERVED))
def test_our_ids_equal_transformers_ids(name):
    _, ours, theirs = checkpoint(name)
    pieces = PromptIds(ours)
    for text in texts(ours):
        want = theirs.encode(text, add_special_tokens=False)
        assert ours.encode(text, add_special_tokens=False).ids == want, repr(text[:80])     # a raw prompt
        assert pieces.encode(text) == want, repr(text[:80])                                 # a chat prompt


@pytest.mark.parametrize("name", sorted(SERVED))
def test_decode_gives_the_text_back(name):
    _, ours, theirs = checkpoint(name)
    for text in texts(ours):
        ids = ours.encode(text, add_special_tokens=False).ids
        back = ours.decode(ids, skip_special_tokens=False)
        assert back == normalized(ours, text), repr(text[:80])
        assert theirs.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == back


@pytest.mark.parametrize("name", sorted(SERVED))
def test_tokenize_adds_the_special_tokens_transformers_adds(name):
    """vLLM's /tokenize encodes ``prompt`` with add_special_tokens true by default, through transformers."""

    _, ours, theirs = checkpoint(name)
    for text in ("Hello world", "日本語 🙂", added_text(ours)):
        assert ours.encode(text, add_special_tokens=True).ids == theirs.encode(text, add_special_tokens=True), text


# -- BOS ---------------------------------------------------------------------------------------------------------

def app(name: str):
    from tests.test_prompt_parity import cuda_app

    folder, _, _ = checkpoint(name)
    return cuda_app(name, folder, True, None)     # GlmApp for "glm", Lfm2MoeApp for "lfm2", else App


def bos_ids(name: str) -> dict[int, str]:
    """BOS as the tokenizer, config.json and generation_config.json name it, and the template's own prefix tokens."""

    folder, ours, theirs = checkpoint(name)
    ids: dict[int, str] = {}
    for token in (*SERVED[name][1], theirs.bos_token):
        if token and ours.token_to_id(token) is not None:
            ids[ours.token_to_id(token)] = token
    for file in ("config.json", "generation_config.json"):
        path = folder / file
        value = json.loads(path.read_text()).get("bos_token_id") if path.is_file() else None
        if isinstance(value, int) and value not in ids:
            ids[value] = ours.id_to_token(value)
    return ids


@pytest.mark.parametrize("name", sorted(SERVED))
def test_a_chat_prompt_holds_one_bos_at_most(name):
    from tests.test_prompt_parity import CONVERSATIONS, Mac

    folder, ours, _ = checkpoint(name)
    served, marks = app(name), bos_ids(name)
    hf = Mac(name, folder, True, None)          # transformers' apply_chat_template, as the parity test calls it
    for label, (messages, tools) in CONVERSATIONS.items():
        body = {"messages": messages, **({"tools": tools} if tools else {})}
        prompt = served._prepare(body, True).prompt
        reference = hf.prompt(body, messages, tools)
        text = ours.decode(prompt, skip_special_tokens=False)
        for token_id, token in marks.items():
            assert prompt.count(token_id) <= 1, (label, token)
            assert prompt.count(token_id) == reference.count(token_id), (label, token)
            # the rendered text sent back as a raw prompt, or to /tokenize, gets no second copy
            raw = served._prepare({"prompt": text}, False).prompt
            assert raw.count(token_id) == prompt.count(token_id), (label, token)
            assert served.tokenize({"prompt": text})["tokens"].count(token_id) == prompt.count(token_id), (label, token)
        if SERVED[name][1]:          # a template that writes its prefix writes it first, once
            assert ours.id_to_token(prompt[0]) == SERVED[name][1][0], label


class Recorder:
    """An engine that keeps each prompt it is handed and ends the reply at once."""

    eos, tp = (0,), 1

    def __init__(self):
        self.prompts = []
        self.request = SimpleNamespace()      # GlmApp.run sets the request's policy and stop_eos here

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        self.prompts.append(list(prompt))
        on_tokens([0])
        return {"rounds": 1}


@pytest.mark.parametrize("name", sorted(SERVED))
def test_the_engine_is_handed_the_prompt_as_rendered(name):
    """A chat request reaches the engine with ``_prepare``'s ids: nothing adds a BOS between render and generate."""

    from tensorfold import families
    from tensorfold.cuda.server import App

    folder, _, _ = checkpoint(name)
    engine = Recorder()
    cls = getattr(families.detect(folder).package, "CUDA_APP", None) or App
    served = cls(engine, folder, name, default_thinking=True)
    body = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}
    want = served._prepare(body, True).prompt
    served.run(body, True, lambda delta: True)
    assert engine.prompts == [want]
    assert all(want.count(token_id) <= 1 for token_id in bos_ids(name))
