"""The grammar's vocabulary width from config.json: text_config, llm_config, then the top level (upstream #951, #953)."""

import json
import os
from pathlib import Path

import pytest

from tensorfold.engine.grammar import vocab_size

BOLT = Path(os.environ.get("MODELS", "/home/agis/models")) / "Nemotron-3.5-Bolt-74B-A9B-24GB"


def _vocab(tmp_path, config) -> int | None:
    (tmp_path / "config.json").write_text(json.dumps(config))
    return vocab_size(tmp_path)


@pytest.mark.skipif(not (BOLT / "config.json").is_file(), reason="Nemotron Bolt checkpoint not present")
def test_nemotron_bolt_reads_its_text_model_s_vocabulary():
    assert json.loads((BOLT / "config.json").read_text())["model_type"] == "nemotron_h_omni"
    assert vocab_size(BOLT) == 131072


@pytest.mark.parametrize("config, expected", [
    ({"model_type": "nemotron_h_omni", "llm_config": {"model_type": "nemotron_h", "vocab_size": 131072}}, 131072),
    ({"vocab_size": 32000}, 32000),
    ({"text_config": {"vocab_size": 32001}, "vocab_size": 32000}, 32001),
    ({"text_config": {}, "vocab_size": 32000}, 32000),
    ({"text_config": None, "vocab_size": 32000}, 32000),
    ({"llm_config": {"vocab_size": 131072}, "vocab_size": 32000}, 131072),
    ({"llm_config": None, "vocab_size": 32000}, 32000),
    ({"text_config": {"vocab_size": 32001}, "llm_config": {"vocab_size": 131072}, "vocab_size": 32000}, 32001),
    ({"text_config": {"vocab_size": -1}, "llm_config": {"vocab_size": 131072}}, 131072),
    ({"text_config": {"vocab_size": True}, "vocab_size": 32000}, 32000),
    ({"text_config": {"vocab_size": 2**31}, "llm_config": {"vocab_size": 131072}}, 131072),
    ({"vocab_size": 2**31 - 1}, 2**31 - 1),
    ({"vocab_size": 2**31}, None),
    ({}, None),
    ({"vocab_size": -1}, None),
    ({"vocab_size": 0}, None),
    ({"vocab_size": True}, None),
    ({"vocab_size": "131072"}, None),
    ({"vocab_size": 131072.0}, None),
    ({"text_config": [1], "llm_config": "x", "vocab_size": 32000}, 32000),
])
def test_the_text_vocabulary_wins_and_bad_values_fall_through(tmp_path, config, expected):
    assert _vocab(tmp_path, config) == expected


def test_no_config_has_no_vocabulary(tmp_path):
    assert vocab_size(tmp_path) is None
