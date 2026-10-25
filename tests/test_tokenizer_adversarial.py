"""The ids the CUDA server encodes (``PromptIds`` over the ``tokenizers`` Tokenizer it loads from tokenizer.json) equal
transformers' AutoTokenizer ids on adversarial text, and decode back to the text, on all nine served checkpoints. A
chat prompt rendered by ``ChatTemplate.render`` holds the checkpoint's BOS at most once, at the start.

``tests/test_tokenizer_parity.py`` checks llama.cpp's ``CHK_TXT`` set on five of the checkpoints. This file adds Super
VL, clef-flash, d1-3B and Gemma 4, and the texts that set lacks: NFKC bait, lone CR, U+FFFD, private-use and control
characters, bidi marks, astral letters, a long word, a long digit run and near-spellings of each added token. A
missing checkpoint is skipped (playbook 8.1, 8.2)."""

from __future__ import annotations

import functools
import json
import os
import unicodedata
from pathlib import Path

import pytest

pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")
transformers = pytest.importorskip("transformers")

from tokenizers import Tokenizer

from tensorfold.cuda.chat_template import ChatTemplate
from tensorfold.cuda.prompt_ids import PromptIds

MODELS = Path(os.environ.get("MODELS", str(Path.home() / "models")))
SERVED = ["Qwen3.8-Flash-Next-exl3-3.05bpw", "Qwen3.8-Flash-Next-FP4-Local", "glm53-flash-sushi-experts-mosaic-dense",
          "Nemotron-3.5-Bolt-74B-A9B-24GB", "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4", "LFM2.5-24B-A2B-FP8",
          "clef-flash", "d1-3B", "gemma-4-31B-it-qat-w4a16-ct"]
CAFE = "café naïve Ångström ﬁ ½ Ⅻ"
ADVERSARIAL = {
    "mixed scripts": "English Ελληνικά русский 日本語 한국어 العربية עברית हिन\u094dदी ไทย",
    "combining marks": "e\u0301 a\u0308\u0304 Z\u0335\u0321\u0353a\u0336l\u0337g\u0338o\u0489 क\u094dष",
    "emoji zwj": "👩\u200d👩\u200d👧\u200d👦 🏳\ufe0f\u200d🌈 👨🏽\u200d💻 ❤\ufe0f\u200d🔥 🧑\u200d🤝\u200d🧑 1\ufe0f\u20e3 🇬🇷🇯🇵",
    "nfc": unicodedata.normalize("NFC", CAFE),
    "nfd": unicodedata.normalize("NFD", CAFE),
    "nfkc bait": "ﬁ ① ｆｕｌｌ ㎏ ℌ ²",
    "crlf": "line one\r\nline two\r\n\r\nend\r\n",
    "lone cr": "a\rb\r\rc\r",
    "tabs and spaces": "\tx\t\ty" + " " * 37 + "z  \t \n    \n\t\n",
    "long word": "Pneumonoultramicroscopicsilicovolcanoconiosis" * 40,
    "digits": "3141592653589793238462643383279502884197169399375105820974944592" * 8 + " 1,000,000.00 0x7f 1e-9",
    "replacement char": "\ufffd bad \ufffd\ufffd bytes",
    "private use": "\ue000\uf8ff \U000f0000\U0010fffd x\ue123y",
    "controls": "\x00\x01\x07\x1b[31mred\x1b[0m\x7f\u200b\u200c\u200d\u2060\ufeff\u00ad",
    "bidi marks": "abc\u202eفارسی\u202c\u200f\u200e123",
    "astral": "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𓀀 𠜎𠜱 🀄",
}
# clef-flash's tokenizer.json keeps Qwen2's split regex (\p{L}+). Its tokenizer_config.json names Qwen3.5's
# ([\p{L}\p{M}]+) as pretokenize_regex, which transformers 5 builds and the server does not read, so the server splits
# a letter from its combining mark. Flash Next's tokenizer.json has the new regex
CLEF_REGEX = pytest.mark.xfail(strict=True, reason=(
    "mismatch: clef-flash's tokenizer.json pre-tokenizer splits letters from combining marks; transformers applies "
    "tokenizer_config.json's pretokenize_regex (क\u094dष: server [62516, 157451], transformers [151858])"))


@functools.cache
def served(name: str):
    folder = MODELS / name
    if not (folder / "tokenizer.json").exists():
        pytest.skip(f"{folder} is not here")
    tok = Tokenizer.from_file(str(folder / "tokenizer.json"))         # what cuda.server.App loads
    return tok, PromptIds(tok)


def near_spellings(tok: Tokenizer) -> str:
    """Each added token cut, spaced, upper-cased, with full-width brackets, glued to letters and doubled. Of Bolt's
    1,000, the named ones come before its <SPECIAL_n> placeholders, up to 100."""

    names = sorted({str(t.content) for t in tok.get_added_tokens_decoder().values()},
                   key=lambda n: (n.startswith("<SPECIAL_"), n))[:100]
    out = []
    for n in names:
        out += [n[:-1], n[1:], n[0] + " " + n[1:], n.upper(), n.replace("_", "-"), n.replace("<", "＜"),
                n.replace("|", "｜"), f"x{n}y", f" {n} ", n + n]
    return " ".join(out)


def texts(tok: Tokenizer) -> dict[str, str]:
    return {**ADVERSARIAL, "near added": near_spellings(tok)}


@pytest.mark.parametrize("name", [pytest.param(n, marks=CLEF_REGEX) if n == "clef-flash" else n for n in SERVED])
def test_the_servers_ids_equal_transformers_ids(name):
    tok, prompts = served(name)
    hf = transformers.AutoTokenizer.from_pretrained(str(MODELS / name), local_files_only=True)
    misses = {}
    for kind, text in texts(tok).items():
        ours, theirs = prompts.encode(text), hf(text, add_special_tokens=False).input_ids
        if ours != theirs:
            misses[kind] = next(i for i, (a, b) in enumerate(zip([*ours, None], [*theirs, None])) if a != b)
    assert misses == {}, f"{name}: first differing id per text {misses}"


@pytest.mark.parametrize("name", SERVED)
def test_adversarial_text_decodes_back_to_itself(name):
    tok, prompts = served(name)
    normalizer = json.loads(tok.to_str())["normalizer"]
    if normalizer is None or normalizer["type"] == "Replace":      # Gemma's " " to "▁", which its decoder undoes
        same = str
    elif normalizer["type"] == "NFC":                              # Qwen's and clef's: lossless up to NFC
        same = functools.partial(unicodedata.normalize, "NFC")
    else:
        raise AssertionError(f"{name}: no round-trip rule for the {normalizer['type']} normalizer")
    lossy = {}
    for kind, text in texts(tok).items():
        back = tok.decode(prompts.encode(text), skip_special_tokens=False)
        if back != same(text):
            lossy[kind] = back
    assert lossy == {}, f"{name}: {json.dumps(lossy, ensure_ascii=False)[:600]}"


def chat(bos: str) -> list[dict]:
    return [{"role": "system", "content": f"{bos}You are terse."},
            {"role": "user", "content": f"{bos}Hi {bos}"}, {"role": "assistant", "content": "Hello."},
            {"role": "user", "content": "Again?"}]


@pytest.mark.parametrize("name", SERVED)
@pytest.mark.parametrize("thinking", [True, False])
def test_a_rendered_prompt_holds_the_bos_once_at_the_start(name, thinking):
    tok, prompts = served(name)
    template = ChatTemplate(MODELS / name)
    bos = template.specials.get("bos_token")
    bos_id = tok.token_to_id(bos) if bos else None
    if bos_id is None:
        pytest.skip(f"{name} has no BOS token")
    for messages in (chat(""), chat(bos)):            # the request spelling the BOS encodes as plain text
        text = template.render(messages, tools=None, enable_thinking=thinking, request=prompts.request)
        ids = prompts.encode(text)
        assert ids.count(bos_id) == text.startswith(bos)    # Bolt and Super VL write none; LFM2, d1 and Gemma one
        assert bos_id not in ids or ids[0] == bos_id
