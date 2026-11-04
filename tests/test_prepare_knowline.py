"""prepare.sh's ``knowline``: after the download, a serving folder of links to the release plus label_readout.json,
which run.sh's guard check and the server's label readout both read. Runs the function with fetch stubbed."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "prepare.sh").read_text()


def test_the_serving_folder_links_the_release_and_names_its_layout(tmp_path):
    models = tmp_path / "models"
    release = models / "KnowLine-4B-Gen5.5"
    release.mkdir(parents=True)
    for name in ("config.json", "model-00001-of-00002.safetensors", "tokenizer.json"):
        (release / name).write_text(name)
    script = "\n".join([
        "set -euo pipefail", f"MODELS={models}",
        re.search(r"^KNOWLINE=\(.*\)$", TEXT, flags=re.M).group(0),
        "fetch() { :; }",
        re.search(r"^knowline\(\) {.*?^}$", TEXT, flags=re.S | re.M).group(0),
        "knowline", "knowline",                               # a rerun leaves the same folder
    ])
    run = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    view = models / "KnowLine-4B-Gen5.5-serve"
    assert json.loads((view / "label_readout.json").read_text()) == {"layout": "knowline", "read_twice": True}
    assert (view / "config.json").is_symlink() and (view / "config.json").read_text() == "config.json"
    assert sorted(p.name for p in view.iterdir()) == ["config.json", "label_readout.json",
                                                      "model-00001-of-00002.safetensors", "tokenizer.json"]
    assert re.search(r"^KNOWLINE=\(PelaAI/KnowLine-4B-Gen5\.5 [0-9a-f]{40} ", TEXT, flags=re.M)
