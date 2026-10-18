"""Each served checkpoint's CUDA family stops on every end token the checkpoint names: config.json's (top level and
text_config), generation_config.json's, and the tokenizer's chat end token (CPU only).

The 0.3.6.2 changelog records what a missing one costs: tool calls and think tags leaked into replies. The real
checkpoints come from $MODELS; a missing one is skipped. Each case also runs on a copy of the checkpoint's config with
one more end id in generation_config.json only, then in config.json only, so a reader that skips a file fails here
even where today's checkpoint lists the same ids in both."""

from __future__ import annotations

import importlib
import json
import os
import shutil
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")          # Flash Next's and GLM's CUDA weights import their triton kernels
pytest.importorskip("tokenizers")
pytest.importorskip("jinja2")

MODELS = Path(os.environ.get("MODELS", str(Path.home() / "models")))
# name: (folder under $MODELS, the module whose Config.read gives the engine's end ids)
SERVED = {
    "flashnext-exl3": ("Qwen3.8-Flash-Next-exl3-3.05bpw", "tensorfold.families.qwen4_exp.cuda.weights"),
    "flashnext-fp4": ("Qwen3.8-Flash-Next-FP4-Local", "tensorfold.families.qwen4_exp.cuda.weights"),
    "glm": ("glm53-flash-sushi-experts-mosaic-dense", "tensorfold.families.glm5_next.cuda.weights"),
    "bolt": ("Nemotron-3.5-Bolt-74B-A9B-24GB", "tensorfold.families.nemotron_bolt.cuda.weights"),
    "lfm2": ("LFM2.5-24B-A2B-FP8", "tensorfold.families.lfm2_moe.config"),
}

def folder_of(name: str) -> Path:
    folder = MODELS / SERVED[name][0]
    if not (folder / "config.json").is_file():
        pytest.skip(f"{folder} is not here")
    return folder


def family_eos(name: str, folder: Path) -> tuple[int, ...]:
    return tuple(importlib.import_module(SERVED[name][1]).Config.read(folder).eos)


def ids(value) -> list[int]:
    return [int(v) for v in value] if isinstance(value, list) else [] if value is None else [int(value)]


def config_ids(folder: Path) -> list[int]:
    raw = json.loads((folder / "config.json").read_text())
    return ids(raw.get("eos_token_id")) + ids((raw.get("text_config") or {}).get("eos_token_id"))


def generation_ids(folder: Path) -> list[int]:
    path = folder / "generation_config.json"
    return ids(json.loads(path.read_text()).get("eos_token_id")) if path.is_file() else []


def chat_end_ids(folder: Path) -> list[int]:
    """The tokenizer's eos_token, and the added token the chat template writes after an assistant turn."""

    from tokenizers import Tokenizer

    from tensorfold.cuda.chat_template import ChatTemplate

    tok = Tokenizer.from_file(str(folder / "tokenizer.json"))
    found = []
    eos = json.loads((folder / "tokenizer_config.json").read_text()).get("eos_token")
    eos = eos.get("content") if isinstance(eos, dict) else eos
    if eos and tok.token_to_id(eos) is not None:
        found.append(tok.token_to_id(eos))
    text = ChatTemplate(folder).render([{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "MARK"}],
                                       tools=None, enable_thinking=False, extra={"add_generation_prompt": False})
    after = tok.encode(text[text.rindex("MARK") + len("MARK"):], add_special_tokens=False).ids
    added = set(tok.get_added_tokens_decoder())
    turn_end = next((t for t in after if t in added), None)
    if turn_end is not None:
        found.append(turn_end)
    return found


def cases():
    return sorted(SERVED)


@pytest.mark.parametrize("name", sorted(SERVED))
def test_the_family_stops_on_every_end_token_the_checkpoint_names(name):
    folder = folder_of(name)
    want = set(config_ids(folder)) | set(generation_ids(folder)) | set(chat_end_ids(folder))
    got = family_eos(name, folder)
    assert len(got) == len(set(got)) and set(got) == want, (sorted(got), sorted(want))


def copy(folder: Path, into: Path, config_eos, generation_eos) -> Path:
    """The checkpoint's small files with the end ids replaced: ``None`` drops the key."""

    for file in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (folder / file).is_file():
            shutil.copy(folder / file, into / file)
    raw = json.loads((folder / "config.json").read_text())
    for part in (raw, raw.get("text_config") or {}):
        part.pop("eos_token_id", None)
    if config_eos is not None:
        (raw.get("text_config") or raw)["eos_token_id"] = config_eos
    (into / "config.json").write_text(json.dumps(raw))
    gen = json.loads((folder / "generation_config.json").read_text())
    gen.pop("eos_token_id", None)
    if generation_eos is not None:
        gen["eos_token_id"] = generation_eos
    (into / "generation_config.json").write_text(json.dumps(gen))
    return into


def known_and_spare(folder: Path) -> tuple[list[int], int]:
    """The checkpoint's end ids, and an id beyond them to place in one file only."""

    every = list(dict.fromkeys(config_ids(folder) + generation_ids(folder)))
    return every, max(every) + 1


@pytest.mark.parametrize("name", cases())
def test_an_end_id_only_in_generation_config_is_kept(name, tmp_path):
    folder = folder_of(name)
    every, spare = known_and_spare(folder)
    got = family_eos(name, copy(folder, tmp_path, every, [*every, spare]))
    assert spare in got and set(every) <= set(got)


@pytest.mark.parametrize("name", sorted(SERVED))
def test_an_end_id_only_in_config_is_kept(name, tmp_path):
    folder = folder_of(name)
    every, spare = known_and_spare(folder)
    got = family_eos(name, copy(folder, tmp_path, [*every, spare], every))
    assert spare in got and set(every) <= set(got)
