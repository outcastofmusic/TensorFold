#!/usr/bin/env bash
# Serve one model from this checkout on a DGX Spark, with the settings the fork measured as best (README, "Run it").
#
#   ./run.sh qwen-flash-next   Qwen3.8 Flash Next on 127.0.0.1:30000, with Clef-Flash beside it as the thinking guard
#   ./run.sh glm-5.3-flash     GLM-5.3 Flash on 127.0.0.1:30000, with its DFlash2 draft model
#   ./run.sh stop              stop whatever this script started
#   ./run.sh status            what is running and whether it answers
#
# The two models do not fit together: starting one stops the other (about 1 minute to Qwen, 1.5 to GLM). Run ./prepare.sh first.
# The server listens on loopback only and checks no API key; put a gateway in front of it to reach it from elsewhere.
#
# Settings (environment):
#   MODELS=~/models            where prepare.sh put the checkpoints
#   PORT=30000                 the model's port;  CLEF_PORT=30001 the guard's
#   TF_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
#   EXTRA="..."                more `tensorfold serve` flags
#   QWEN_MODEL, CLEF_MODEL, GLM_MODEL   another checkpoint folder under $MODELS
# Qwen:  CONTEXT=262144  PARALLEL=4  KV=int8  THINK_BUDGET=8000 (0: none)  GUARD=1 (0: no Clef)  VISION=1 (0: text only)
# GLM:   CONTEXT=98304   DRAFTER=cfontes (modal: the MIT one; none: no draft model, then CONTEXT=131072 fits)
set -euo pipefail

MODELS="${MODELS:-$HOME/models}"
BASE="${TF_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
PORT="${PORT:-30000}"
CLEF_PORT="${CLEF_PORT:-30001}"
SRC="$(cd "$(dirname "$0")" && pwd)"
NAMES="tensorfold-serve clef-serve tensorfold-glm"

die() { echo "run: $*" >&2; exit 1; }
up() { docker ps --format '{{.Names}}' | grep -qx "$1"; }
avail() { awk '/MemAvailable/{printf "%d", $2 / 1048576}' /proc/meminfo; }
avail_mib() { awk '/MemAvailable/{printf "%d", $2 / 1024}' /proc/meminfo; }

stop_all() {
    local was=0 n i prev now
    for n in $NAMES; do
        up "$n" && { docker stop -t 5 "$n" >/dev/null; was=1; echo "run: stopped $n"; }
        docker rm -f "$n" >/dev/null 2>&1 || true
    done
    [ "$was" = 1 ] || return 0
    # The stopped server's memory comes back a few seconds after the stop returns: wait for it to start rising (10 s at
    # most), then for 2 s in a row without another 256 MiB.
    local rose=0 flat=0
    prev=$(avail_mib)
    for i in $(seq 60); do
        sleep 1
        now=$(avail_mib)
        if [ $((now - prev)) -ge 256 ]; then rose=1; flat=0; else flat=$((flat + 1)); fi
        prev=$now
        { [ "$rose" = 1 ] && [ "$flat" -ge 2 ]; } || { [ "$rose" = 0 ] && [ "$i" -ge 10 ]; } && break
    done
}

# The checkout installed into a local image, rebuilt when a source file changes. A boot from it skips pip, and the
# CUDA extensions compiled on its first boot stay in the tensorfold-ext volume: 100 s of compiling a boot otherwise.
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
        -v "$MODELS:$MODELS:ro" -v tensorfold-ext:/ext -e TORCH_EXTENSIONS_DIR=/ext -e HF_HUB_OFFLINE=1 "${flags[@]}" "$IMAGE" \
        tensorfold serve "$@" --no-update-check >/dev/null
}

wait_ready() {  # container name, port; returns 1 when the server exited while loading
    local i
    echo "run: $1 is loading; docker logs -f $1"
    for i in $(seq 450); do
        curl -sf -m 3 "http://127.0.0.1:$2/v1/models" >/dev/null 2>&1 && { echo "run: $1 answers on http://127.0.0.1:$2/v1"; return 0; }
        up "$1" || return 1
        sleep 2
    done
    die "$1 did not answer within 15 minutes"
}

failed() { docker logs "$1" 2>&1 | tail -5 >&2; die "$1 exited while loading"; }

qwen() {
    local model="${QWEN_MODEL:-$MODELS/Qwen3.8-Flash-Next-exl3-3.05bpw}" clef="${CLEF_MODEL:-$MODELS/clef-flash}"
    local tower="$MODELS/Qwen3.8-Flash-Next-exl3-3.05bpw-vision/vision-f16-v3.safetensors"
    [ -e "$model/config.json" ] || die "no checkpoint at $model (run ./prepare.sh qwen-flash-next)"
    IMAGE=$(image)
    stop_all
    local guard=() budget=() vision=() flags=()
    if [ "${GUARD:-1}" != 0 ]; then
        [ -e "$clef/joint_head.safetensors" ] || die "no Clef-Flash at $clef (run ./prepare.sh qwen-flash-next, or GUARD=0)"
        serve clef-serve -- "$clef" --host 127.0.0.1 --port "$CLEF_PORT" --name clef --context 16384
        guard=(--thinking-guard "http://127.0.0.1:$CLEF_PORT/v1")
    fi
    [ "${THINK_BUDGET:-8000}" = 0 ] || budget=(--thinking-budget "${THINK_BUDGET:-8000}")
    if [ "${VISION:-1}" != 0 ] && [ -e "$tower" ]; then
        vision=(--vision); flags=(-e "TENSORFOLD_VISION_WEIGHTS=$tower")
    fi
    serve tensorfold-serve "${flags[@]}" -- "$model" --host 127.0.0.1 --port "$PORT" --name qwen3.8-flash-next \
        --context "${CONTEXT:-262144}" --max-tokens "${CONTEXT:-262144}" --parallel "${PARALLEL:-4}" --kv-dtype "${KV:-int8}" \
        "${vision[@]}" "${guard[@]}" "${budget[@]}" ${EXTRA:-}
    [ "${GUARD:-1}" = 0 ] || wait_ready clef-serve "$CLEF_PORT" || failed clef-serve
    wait_ready tensorfold-serve "$PORT" || failed tensorfold-serve
}

glm() {
    local model="${GLM_MODEL:-$MODELS/glm53-flash-mosaic-12l}" drafter=() context=98304
    [ -e "$model/config.json" ] || die "no checkpoint at $model (run ./prepare.sh glm-5.3-flash)"
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
    # GLM fits with 0.5 GiB to spare, so a boot can be refused while the last of the memory returns: try again, up to 4 times.
    local try
    for try in 1 2 3 4; do
        docker rm -f tensorfold-glm >/dev/null 2>&1 || true
        # TF_GLM_DENSE_PROMPT=1: prompt rows through the dense matmul, 310 -> 362 tok/s on a 20,500-token prompt
        serve tensorfold-glm -e TF_GLM_DENSE_PROMPT=1 -- "$model" --tp 1 --host 127.0.0.1 --port "$PORT" --name glm-5.3-flash \
            --context "${CONTEXT:-$context}" --max-tokens "${CONTEXT:-$context}" "${drafter[@]}" ${EXTRA:-}
        wait_ready tensorfold-glm "$PORT" && return 0
        docker logs tensorfold-glm 2>&1 | grep -q 'memory budget cannot fit' || break
        echo "run: refused for lack of memory ($(avail) GiB available), trying again in 10 s" >&2
        sleep 10
    done
    failed tensorfold-glm
}

status() {
    local n any=0
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
    qwen-flash-next) qwen ;;
    glm-5.3-flash)   glm ;;
    stop)            stop_all ;;
    status)          status ;;
    *) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
