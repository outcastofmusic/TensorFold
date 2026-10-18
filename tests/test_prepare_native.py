"""prepare.sh's native server install (``super``): the archive unpacks into a temp folder under $MODELS and moves into
place only once its binary is there, so a failed or interrupted unpack leaves neither a folder a rerun would skip nor
the temp folder. Runs the function from prepare.sh with curl and fetch stubbed."""

from __future__ import annotations

import hashlib
import io
import re
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.12"
FOLDER = f"tensorfold-{VERSION}-linux-aarch64"


def _function(name: str) -> str:
    text = (ROOT / "prepare.sh").read_text()
    return re.search(rf"^{name}\(\) {{.*?^}}$", text, flags=re.S | re.M).group(0)


def _archive(path: Path, *, binary: bool = True) -> str:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        entries = [(f"{FOLDER}/README", 0o644)] + ([(f"{FOLDER}/bin/tensorfold-native", 0o755)] if binary else [])
        for name, mode in entries:
            info = tarfile.TarInfo(name)
            info.size, info.mode = 4, mode
            tar.addfile(info, io.BytesIO(b"data"))
    path.write_bytes(buf.getvalue())
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _install(tmp_path: Path, archive: Path, sha: str, *, tar: str = "") -> subprocess.CompletedProcess:
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    script = "\n".join([
        "set -euo pipefail",
        f"MODELS={models}",
        f"NATIVE=({VERSION} {sha})",
        "SUPER=()",
        re.search(r"^die\(\) .*$", (ROOT / "prepare.sh").read_text(), flags=re.M).group(0),
        "fetch() { :; }",
        f'curl() {{ while [ "$1" != -o ]; do shift; done; cp {archive} "$2"; }}',
        tar,
        _function("super"),
        "super",
    ])
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)


def _left(tmp_path: Path) -> list[str]:
    return sorted(p.name for p in (tmp_path / "models").iterdir())


def test_a_good_archive_lands_in_its_folder_and_no_temp_folder_stays(tmp_path):
    sha = _archive(tmp_path / "a.tar.gz")
    run = _install(tmp_path, tmp_path / "a.tar.gz", sha)
    assert run.returncode == 0, run.stderr
    assert (tmp_path / "models" / FOLDER / "bin" / "tensorfold-native").stat().st_mode & 0o111
    assert _left(tmp_path) == [FOLDER]


def test_a_broken_earlier_unpack_is_replaced(tmp_path):
    (tmp_path / "models" / FOLDER / "bin").mkdir(parents=True)          # an interrupted unpack by the old prepare.sh
    sha = _archive(tmp_path / "a.tar.gz")
    run = _install(tmp_path, tmp_path / "a.tar.gz", sha)
    assert run.returncode == 0, run.stderr
    assert (tmp_path / "models" / FOLDER / "bin" / "tensorfold-native").is_file()
    assert _left(tmp_path) == [FOLDER]


@pytest.mark.parametrize("case", ["no binary", "interrupted unpack", "wrong sha256"])
def test_a_failed_install_leaves_neither_the_folder_nor_the_temp_folder(tmp_path, case):
    sha = _archive(tmp_path / "a.tar.gz", binary=case != "no binary")
    tar = 'tar() { command tar "$@"; return 1; }' if case == "interrupted unpack" else ""
    run = _install(tmp_path, tmp_path / "a.tar.gz", "0" * 64 if case == "wrong sha256" else sha, tar=tar)
    assert run.returncode == 1 and "prepare: " in run.stderr, run.stderr
    assert _left(tmp_path) == []
