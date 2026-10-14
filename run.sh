#!/usr/bin/env bash
# Serve one model from this checkout on a DGX Spark, with the settings the fork measured as best (README, "Run it").
#
#   ./run.sh qwen-flash-next   Qwen3.8 Flash Next on 127.0.0.1:30000, with Clef-Flash beside it as the thinking guard
#   ./run.sh glm-5.3-flash     GLM-5.3 Flash on 127.0.0.1:30000, with its DFlash2 draft model
#   ./run.sh swap              one endpoint on 127.0.0.1:30000 that loads whichever model a request names (swap.py)
#   ./run.sh stop              stop whatever this script started
#   ./run.sh status            what is running and whether it answers
#
# The two models do not fit together: starting one stops the other (21 to 24 s to either). Run ./prepare.sh first.
# The server listens on loopback only and checks no API key; put a gateway in front of it to reach it from elsewhere.
#
# Settings (environment):
#   MODELS=~/models            where prepare.sh put the checkpoints
#   PORT=30000                 the model's port;  CLEF_PORT=30001 the guard's;  BACKEND_PORT=30002 the model's under swap
#   TF_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
#   EXTRA="..."                more `tensorfold serve` flags
#   QWEN_MODEL, CLEF_MODEL, GLM_MODEL   another checkpoint folder under $MODELS
# Qwen:  CONTEXT=262144  PARALLEL=4  KV=int8  THINK_BUDGET=8000 (0: none)  GUARD=1 (0: no Clef)  VISION=1 (0: text only)
# GLM:   CONTEXT=98304   DRAFTER=cfontes (modal: the MIT one; none: no draft model, then CONTEXT=131072 fits)
#        TIMING=0 (1: time the draft steps at start, which the `auto:...` draft chooser needs)
set -euo pipefail

MODELS="${MODELS:-$HOME/models}"
BASE="${TF_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
PORT="${PORT:-30000}"
CLEF_PORT="${CLEF_PORT:-30001}"
SRC="$(cd "$(dirname "$0")" && pwd)"
NAMES="tensorfold-serve clef-serve tensorfold-glm"
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/tensorfold-run"

die() { echo "run: $*" >&2; exit 1; }
up() { docker ps --format '{{.Names}}' | grep -qx "$1"; }
avail() { awk '/MemAvailable/{printf "%d", $2 / 1048576}' /proc/meminfo; }
avail_mib() { awk '/MemAvailable/{printf "%d", $2 / 1024}' /proc/meminfo; }

uncache() {  # drop the served checkpoints' files from the page cache (they stay on disk)
    local f
    for f in "$MODELS"/{Qwen3.8-Flash-Next-exl3-3.05bpw,clef-flash,glm53-flash-mosaic-12l,glm53-flash-sushi-experts-mosaic-dense,GLM-5.3-Flash-DFlash*}/*.safetensors \
             "${QWEN_MODEL:-}"/*.safetensors "${CLEF_MODEL:-}"/*.safetensors "${GLM_MODEL:-}"/*.safetensors; do
        [ -f "$f" ] && dd if="$f" iflag=nocache count=0 status=none 2>/dev/null
    done
    return 0
}

stop_all() {
    local was=0 n start=$SECONDS
    for n in $NAMES; do
        up "$n" && was=1
        docker rm -f "$n" >/dev/null 2>&1 &
    done
    wait
    # A killed server's memory is back when the kill returns (1.7 s for GLM's 104 GiB), so the next start follows at
    # once. A GLM start that still finds too little is tried again. What the servers read through the page cache
    # (Qwen's 30 GiB n-gram table) is dropped from it here: left in, the next model's load evicts it page by page.
    [ "$was" = 0 ] || uncache
    [ "$was" = 0 ] || echo "run: stopped what was running, $(avail_mib) MiB available after $((SECONDS - start)) s"
}

# The checkout installed into a local image, rebuilt when a source file changes. A boot from it skips pip, and the
# CUDA extensions and Triton kernels compiled on its first boot stay in the tensorfold-ext volume: 100 s of compiling a
# boot otherwise.
image() {
    local tag old
    tag="tensorfold-run:$(cd "$SRC" && find src pyproject.toml -type f -not -name '*.pyc' -print0 | sort -z | xargs -0 sha1sum | sha1sum | cut -c1-12)"
    if ! docker image inspect "$tag" >/dev/null 2>&1; then
        echo "run: installing this checkout into the image $tag (once per source change)" >&2
        docker rm -f tensorfold-build >/dev/null 2>&1 || true
        docker run --name tensorfold-build -v "$SRC:/tf:ro" "$BASE" \
            bash -c "cp -r /tf /tmp/tf && pip install -q '/tmp/tf[vision]' && rm -rf /tmp/tf" >/dev/null
        docker commit tensorfold-build "$tag" >/dev/null
        docker rm tensorfold-build >/dev/null
        for old in $(docker images --format '{{.Repository}}:{{.Tag}}' tensorfold-run); do
            [ "$old" = "$tag" ] || docker rmi "$old" >/dev/null 2>&1 || true
        done
    fi
    echo "$tag"
}

serve() {  # container name, docker flags..., --, tensorfold serve arguments
    local name=$1; shift
    local flags=()
    while [ "$1" != "--" ]; do flags+=("$1"); shift; done
    shift
    docker run -d --init --name "$name" --gpus all --ipc=host --network host \
        -v "$MODELS:$MODELS:ro" -v tensorfold-ext:/ext -e TORCH_EXTENSIONS_DIR=/ext -e TRITON_CACHE_DIR=/ext/triton -e HF_HUB_OFFLINE=1 "${flags[@]}" "$IMAGE" \
        tensorfold serve "$@" --no-update-check >/dev/null
}

wait_ready() {  # container name, port; returns 1 when the server exited while loading
    local i
    echo "run: $1 is loading; docker logs -f $1"
    for i in $(seq 3600); do
        curl -sf -m 3 "http://127.0.0.1:$2/v1/models" >/dev/null 2>&1 && { echo "run: $1 answers on http://127.0.0.1:$2/v1"; return 0; }
        up "$1" || return 1
        sleep 0.25
    done
    die "$1 did not answer within 15 minutes"
}

failed() { docker logs "$1" 2>&1 | tail -5 >&2; die "$1 exited while loading"; }

swapping() {  # the swap endpoint's process is alive
    local pid
    pid=$(cat "$STATE/swap.pid" 2>/dev/null) || return 1
    [ -n "$pid" ] && grep -qs swap.py "/proc/$pid/cmdline"
}

# Under the swap endpoint a model is loaded by asking it, so the endpoint knows what is in memory.
ask_swap() {
    [ -z "${TF_SWAP_CHILD:-}" ] && swapping || return 1
    echo "run: asking the swap endpoint on 127.0.0.1:$PORT for $1"
    curl -sf -m 1200 -X POST "http://127.0.0.1:$PORT/swap/$1" || die "the swap endpoint could not load $1; see $STATE/swap.log"
    echo
}

swap() {
    command -v python3 >/dev/null || die "swap needs python3 on the host"
    swapping && die "the swap endpoint is already running (pid $(cat "$STATE/swap.pid")); ./run.sh stop ends it"
    stop_all
    mkdir -p "$STATE"
    rm -f "$STATE/swap.pid"
    PORT=$PORT BACKEND_PORT="${BACKEND_PORT:-30002}" SWAP_PIDFILE="$STATE/swap.pid" \
        setsid nohup python3 "$SRC/swap.py" >>"$STATE/swap.log" 2>&1 </dev/null &
    sleep 2
    swapping || { tail -3 "$STATE/swap.log" >&2; die "the swap endpoint did not start"; }
    echo "run: swap endpoint on http://127.0.0.1:$PORT/v1 (log: $STATE/swap.log). A request's \"model\" picks qwen3.8-flash-next or glm-5.3-flash."
    [ -z "${1:-}" ] || ask_swap "$1"
}

qwen() {
    local model="${QWEN_MODEL:-$MODELS/Qwen3.8-Flash-Next-exl3-3.05bpw}" clef="${CLEF_MODEL:-$MODELS/clef-flash}"
    local tower="$MODELS/Qwen3.8-Flash-Next-exl3-3.05bpw-vision/vision-f16-v3.safetensors"
    [ -e "$model/config.json" ] || die "no checkpoint at $model (run ./prepare.sh qwen-flash-next)"
    IMAGE=$(image)
    stop_all
    local guard=() budget=() vision=() flags=()
    if [ "${GUARD:-1}" != 0 ]; then
        [ -e "$clef/joint_head.safetensors" ] || die "no Clef-Flash at $clef (run ./prepare.sh qwen-flash-next, or GUARD=0)"
        guard=(--thinking-guard "http://127.0.0.1:$CLEF_PORT/v1")
    fi
    [ "${THINK_BUDGET:-8000}" = 0 ] || budget=(--thinking-budget "${THINK_BUDGET:-8000}")
    if [ "${VISION:-1}" != 0 ] && [ -e "$tower" ]; then
        vision=(--vision); flags=(-e "TENSORFOLD_VISION_WEIGHTS=$tower")
    fi
    serve tensorfold-serve "${flags[@]}" -- "$model" --host 127.0.0.1 --port "$PORT" --name qwen3.8-flash-next \
        --context "${CONTEXT:-262144}" --max-tokens "${CONTEXT:-262144}" --parallel "${PARALLEL:-4}" --kv-dtype "${KV:-int8}" \
        "${vision[@]}" "${guard[@]}" "${budget[@]}" ${EXTRA:-}
    # Clef starts second: it is ready 17 s before Qwen either way, and Qwen's start is the one a switch waits for
    [ "${GUARD:-1}" = 0 ] || serve clef-serve -- "$clef" --host 127.0.0.1 --port "$CLEF_PORT" --name clef --context 16384
    [ "${GUARD:-1}" = 0 ] || wait_ready clef-serve "$CLEF_PORT" || failed clef-serve
    wait_ready tensorfold-serve "$PORT" || failed tensorfold-serve
}

glm() {
    # The hybrid (Sushi's routed experts on Mosaic 12L's dense part) over Mosaic 12L: TrueScore 93.6 against 86.5,
    # paired difference +2.7 [-1.1, +7.3] over 76 scenarios, decode 5-7% slower (2026-10-13)
    local model="${GLM_MODEL:-$MODELS/glm53-flash-sushi-experts-mosaic-dense}" drafter=() context=98304
    [ -e "$model/config.json" ] || die "no checkpoint at $model (run ./prepare.sh glm-5.3-flash, or GLM_MODEL=$MODELS/glm53-flash-mosaic-12l)"
    case "${DRAFTER:-cfontes}" in
        cfontes) drafter=(--drafter "$MODELS/GLM-5.3-Flash-DFlash2-TR3-v3") ;;
        modal)   drafter=(--drafter "$MODELS/GLM-5.3-Flash-DFlash") ;;
        none)    context=131072 ;;
        *)       drafter=(--drafter "$DRAFTER") ;;
    esac
    [ ${#drafter[@]} = 0 ] || [ -e "${drafter[1]}/config.json" ] || die "no draft model at ${drafter[1]} (./prepare.sh glm-5.3-flash, or DRAFTER=none)"
    IMAGE=$(image)
    stop_all
    [ "$(avail)" -ge 100 ] || echo "run: $(avail) GiB available; GLM needs about 100 and the server refuses to start with less" >&2
    # TENSORFOLD_MEMORY_RESERVE_GIB=11: the server keeps a tenth of the memory free by default (12.2 GiB), which GLM at
    # 98,304 tokens fits under only while 119.4 of the Spark's 121.7 GiB are available; 2 of 6 loads were refused at
    # 119.2 and 119.4 (2026-10-13). A loaded GLM leaves 14 to 16 GiB available either way.
    # A boot can still be refused while the last of the memory returns: try again, up to 4 times.
    local try
    for try in 1 2 3 4; do
        docker rm -f tensorfold-glm >/dev/null 2>&1 || true
        # TF_GLM_DENSE_PROMPT=1: prompt rows through the dense matmul, 310 -> 362 tok/s on a 20,500-token prompt
        # TF_GLM_CALIBRATE=0: no 5 s of draft timing at start; only the `auto:...` chooser reads it (TIMING=1 keeps it)
        # TF_GLM_LAZY_SPARSE=1: the 72 long-context graphs are captured at first use (0.1 s each) where start-up took 7 s
        serve tensorfold-glm -e TF_GLM_DENSE_PROMPT=1 -e TF_GLM_CALIBRATE="${TIMING:-0}" -e TF_GLM_LAZY_SPARSE=1 -e TENSORFOLD_MEMORY_RESERVE_GIB=11 -- "$model" --tp 1 --host 127.0.0.1 --port "$PORT" --name glm-5.3-flash \
            --context "${CONTEXT:-$context}" --max-tokens "${CONTEXT:-$context}" "${drafter[@]}" ${EXTRA:-}
        wait_ready tensorfold-glm "$PORT" && return 0
        docker logs tensorfold-glm 2>&1 | grep -q 'memory budget cannot fit' || break
        echo "run: refused for lack of memory ($(avail) GiB available), trying again in 2 s" >&2
        sleep 2
    done
    failed tensorfold-glm
}

status() {
    local n any=0
    swapping && { any=1; echo "swap endpoint: 127.0.0.1:$PORT $(curl -s -m 3 "http://127.0.0.1:$PORT/swap")"; PORT="${BACKEND_PORT:-30002}"; }
    for n in $NAMES; do
        up "$n" || continue
        any=1
        local port=$PORT; [ "$n" = clef-serve ] && port=$CLEF_PORT
        curl -sf -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && echo "$n: ready on 127.0.0.1:$port" || echo "$n: loading"
    done
    [ "$any" = 1 ] || echo "nothing is running"
}

command -v docker >/dev/null || die "docker is not installed"
case "${1:-}" in
    qwen-flash-next) ask_swap "$1" || qwen ;;
    glm-5.3-flash)   ask_swap "$1" || glm ;;
    swap)            swap "${2:-}" ;;
    stop)            swapping && { kill "$(cat "$STATE/swap.pid")"; rm -f "$STATE/swap.pid"; echo "run: stopped the swap endpoint"; }
                     stop_all ;;
    status)          status ;;
    *) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
