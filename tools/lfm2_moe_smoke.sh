#!/usr/bin/env bash
# A served smoke test of LFM2.5-24B-A2B on a test port: a chat reply with reasoning, a tool call, a streamed reply.
#
#   tools/lfm2_moe_smoke.sh            build this checkout into tensorfold-lfm2-smoke:<hash>, serve on 127.0.0.1:31005,
#                                      run the three requests, stop the server
#
# Production listens on 30000-30010 (the gateway scans them), so the test port is 31005. The script starts only when
# MemAvailable holds the model's 30 GiB plus a 12 GiB margin, and kills its server by name when MemAvailable falls
# below 12 GiB or the swap endpoint on :30000 starts a load. The image gets its own tag, so run.sh's image clean-up
# never removes it and this script never removes run.sh's images.
#
# Environment: MODELS=~/models  LFM2_MODEL=$MODELS/LFM2.5-24B-A2B-FP8  PORT=31005
#              TF_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
set -euo pipefail

MODELS="${MODELS:-$HOME/models}"
MODEL="${LFM2_MODEL:-$MODELS/LFM2.5-24B-A2B-FP8}"
PORT="${PORT:-31005}"
BASE="${TF_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
NAME=tensorfold-lfm2-smoke
URL="http://127.0.0.1:$PORT/v1"

die() { echo "smoke: $*" >&2; exit 1; }
avail_gib() { awk '/MemAvailable/{printf "%d", $2 / 1048576}' /proc/meminfo; }

[ -e "$MODEL/config.json" ] || die "no checkpoint at $MODEL (./prepare.sh lfm2-24b)"
[ "$(avail_gib)" -ge 42 ] || die "$(avail_gib) GiB available; the model needs about 30 plus a 12 GiB margin"
curl -sf -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && die "something already answers on :$PORT"
case "$(curl -s -m 2 http://127.0.0.1:30000/swap 2>/dev/null || true)" in
    *'"loading": "'*) die "the swap endpoint on :30000 is loading a model" ;;
esac

tag="tensorfold-lfm2-smoke:$(cd "$SRC" && find src pyproject.toml run-constraints.txt -type f -not -name '*.pyc' \
    -print0 | sort -z | xargs -0 sha1sum | sha1sum | cut -c1-12)"
if ! docker image inspect "$tag" >/dev/null 2>&1; then
    echo "smoke: installing this checkout into $tag"
    docker rm -f "$NAME-build" >/dev/null 2>&1 || true
    docker run --name "$NAME-build" -v "$SRC:/tf:ro" "$BASE" \
        bash -c "cp -r /tf /tmp/tf && pip install -q -c /tmp/tf/run-constraints.txt /tmp/tf && rm -rf /tmp/tf" \
        >/dev/null
    docker commit "$NAME-build" "$tag" >/dev/null
    docker rm "$NAME-build" >/dev/null
fi

stop() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap stop EXIT
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --init --name "$NAME" --gpus all --ipc=host --network host -v "$MODEL:/model:ro" \
    -v tensorfold-ext:/ext -e TORCH_EXTENSIONS_DIR=/ext -e TRITON_CACHE_DIR=/ext/triton -e HF_HUB_OFFLINE=1 "$tag" \
    tensorfold serve /model --host 127.0.0.1 --port "$PORT" --name lfm2.5-24b-a2b --context 16384 \
    --max-tokens 16384 --top-p 1.0 --no-update-check >/dev/null

# the guard: the server goes when memory runs short or production starts a swap
(
    while docker ps --format '{{.Names}}' | grep -qx "$NAME"; do
        if [ "$(avail_gib)" -lt 12 ]; then echo "smoke: MemAvailable under 12 GiB, stopping $NAME" >&2; stop; fi
        case "$(curl -s -m 2 http://127.0.0.1:30000/swap 2>/dev/null || true)" in
            *'"loading": "'*) echo "smoke: a swap started on :30000, stopping $NAME" >&2; stop ;;
        esac
        sleep 2
    done
) &

echo "smoke: $NAME is loading; docker logs -f $NAME"
for _ in $(seq 900); do
    curl -sf -m 2 "$URL/models" >/dev/null 2>&1 && break
    if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
        docker logs "$NAME" 2>&1 | tail -20 >&2
        die "the server exited"
    fi
    sleep 1
done
curl -sf -m 2 "$URL/models" >/dev/null || die "the server did not answer within 15 minutes"

ask() { curl -sf -m 600 "$URL/chat/completions" -H 'Content-Type: application/json' -d "$1"; }
TOOLS='[{"type": "function", "function": {"name": "get_weather", "description": "Current weather in a city",
         "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]'

echo "smoke: chat"
ask '{"model": "lfm2.5-24b-a2b", "max_tokens": 1024, "temperature": 0,
      "messages": [{"role": "user", "content": "In one sentence: why is the sky blue?"}]}' | python3 -c '
import json, sys
r = json.load(sys.stdin)
m, finish = r["choices"][0]["message"], r["choices"][0]["finish_reason"]
print("reasoning:", (m.get("reasoning_content") or "")[:200].replace("\n", " "))
print("content:", m.get("content"), "| finish:", finish, "| usage:", r.get("usage"))
assert m.get("reasoning_content") and m.get("content") and "</think>" not in m["content"], "no reasoning or answer"
assert finish == "stop", finish'

echo "smoke: tool call"
ask '{"model": "lfm2.5-24b-a2b", "max_tokens": 1024, "temperature": 0,
      "messages": [{"role": "user", "content": "What is the weather in Paris right now?"}],
      "tools": '"$TOOLS"'}' \
    | python3 -c '
import json, sys
r = json.load(sys.stdin)
m, finish = r["choices"][0]["message"], r["choices"][0]["finish_reason"]
calls = m.get("tool_calls") or []
print("calls:", calls, "| content:", m.get("content"), "| finish:", finish)
assert finish == "tool_calls" and calls and calls[0]["function"]["name"] == "get_weather", "no get_weather call"
assert "paris" in json.loads(calls[0]["function"]["arguments"])["city"].lower()
assert "<|tool_call_start|>" not in (m.get("content") or "")'

echo "smoke: tool result, then the answer"
ask '{"model": "lfm2.5-24b-a2b", "max_tokens": 1024, "temperature": 0,
      "messages": [{"role": "user", "content": "What is the weather in Paris right now?"},
                   {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                   {"role": "tool", "tool_call_id": "c1", "content": "{\"temp_c\": 18, \"sky\": \"clear\"}"}],
      "tools": '"$TOOLS"'}' \
    | python3 -c '
import json, sys
r = json.load(sys.stdin)
m = r["choices"][0]["message"]
print("content:", m.get("content"))
assert "18" in (m.get("content") or ""), "the answer does not use the tool result"'

echo "smoke: streamed"
curl -sfN -m 600 "$URL/chat/completions" -H 'Content-Type: application/json' -d '{"model": "lfm2.5-24b-a2b",
    "stream": true, "max_tokens": 512, "seed": 7, "messages": [{"role": "user", "content": "Count from 1 to 5."}]}' \
    | python3 -c '
import json, sys
parts = {"reasoning_content": "", "content": ""}
for line in sys.stdin:
    if line.startswith("data: ") and line.strip() != "data: [DONE]":
        delta = json.loads(line[6:])["choices"][0].get("delta") or {}
        for k in parts:
            parts[k] += delta.get(k) or ""
print("streamed content:", parts["content"].strip()[:200])
assert parts["reasoning_content"] and "5" in parts["content"], parts'

docker logs "$NAME" 2>&1 | grep -E "\[tensorfold\] (LFM2|done|serving)" | tail -6
echo "smoke: passed"
