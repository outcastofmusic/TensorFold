"""TENSORFOLD_REQUEST_LOG is created owner-only: the prompts and tool content it records are not readable by others."""

import json
import os
import stat

from tensorfold.server import http as server_http
from tests.test_server_openai_compat import FakeApp, post_json, serve_fake


def test_request_log_is_created_owner_only(tmp_path, monkeypatch):
    log = tmp_path / "requests.jsonl"
    monkeypatch.setattr(server_http, "_REQUEST_LOG", str(log))
    old = os.umask(0o022)                       # the usual umask would leave a new file 0644
    server = serve_fake(FakeApp())
    try:
        for text in ("first", "second"):
            status, _ = post_json(server, "/v1/chat/completions",
                                  {"model": "fake-model", "messages": [{"role": "user", "content": text}]})
            assert status == 200
    finally:
        os.umask(old)
        server.shutdown()
        server.server_close()
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert [r["messages"][0]["content"] for r in rows] == ["first", "second"]     # appended, one request a line
