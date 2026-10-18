#!/usr/bin/env bash
# Serve one model from this checkout on a DGX Spark, with the settings the fork measured as best (README, "Run it").
#
#   ./run.sh                   the default: ./run.sh swap qwen-flash-next, every model on one endpoint, Qwen loaded first
#   ./run.sh qwen-flash-next   Qwen3.8 Flash Next on 127.0.0.1:30000, with Clef-Flash beside it as the thinking guard
#   ./run.sh glm-5.3-flash     GLM-5.3 Flash on 127.0.0.1:30000, with its DFlash2 draft model
#   ./run.sh qwen-flash-next-fp4   Qwen3.8 Flash Next from NVIDIA's FP4-Local checkpoint (QAD NVFP4), Clef-Flash beside it
#   ./run.sh nemotron-bolt     Nemotron 3.5 Bolt 74B-A9B (24 GB) with its LiLiCorr draft model
#   ./run.sh lfm2-24b          Liquid's LFM2.5-24B-A2B (block FP8, 24.75 GB): tool routing and sub-agents
#   ./run.sh nemotron-3.5-super    Nemotron 3.5 Super VL 120B-A12B (NVFP4, 75.1 GB) on upstream's native 1.1.0 server
#                              With D1=1 each model target but GLM also starts d1-3B, Liquid AI's decision model, on :30003 once
#                              the model answers (POST /v1/systemone and /v1/decisions; docs/d1-sidecar.md)
#   ./run.sh swap              one endpoint on 127.0.0.1:30000 that loads whichever model a request names (swap.py)
#   ./run.sh stop              stop whatever this script started
#   ./run.sh status            what is running and whether it answers
#
# The models do not fit together: starting one stops the other. Run ./prepare.sh first.
# The server listens on loopback only and checks no API key; put a gateway in front of it to reach it from elsewhere.
#
# Settings (environment):
#   MODELS=~/models            where prepare.sh put the checkpoints
#   PORT=30000                 the model's port;  CLEF_PORT=30001 the guard's;  BACKEND_PORT=30002 the model's under swap
#   D1=0 (1: start the d1 decision sidecar)  D1_PORT=30003  D1_CONTEXT=4096: the most tokens one decision reads (state and question)
#   SWAP_DRAIN=600  SWAP_HOLD=0  SWAP_TIMEOUT=3600  SWAP_PERSIST=120   swap.py's waits, in seconds (its header says what each does)
#   TF_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
#   EXTRA="..."                more `tensorfold serve` flags
#   QWEN_MODEL, QWEN_FP4_MODEL, CLEF_MODEL, GLM_MODEL, BOLT_MODEL, LFM2_MODEL, SUPER_MODEL, D1_MODEL   another checkpoint folder under $MODELS
# Qwen, Qwen FP4, GLM:  STATES=1 (0: kept states stay in memory only)  STATES_GIB=64: what each model's folder in the tensorfold-states volume holds, every build's states together;  STATES_TTL_H=72: hours a state stays unused;  Qwen: STATES_ALL=0: each slot's shortest and longest state go to disk (1: every kept state)
# Qwen:  CONTEXT=262144  PARALLEL=4  KV=int8  THINK_BUDGET=8000 (0: none)  GUARD=1 (0: no Clef)  VISION=1 (0: text only)
# GLM:   CONTEXT=98304   THINK_BUDGET=8000 (0: none)   DRAFTER=cfontes (modal: the MIT one; none: no draft model, then CONTEXT=131072 fits)
#        TIMING=0 (1: time the draft steps at start, which the `auto:...` draft chooser needs)
# Qwen FP4-Local: Qwen's settings, VISION=0 by default (the image tower run.sh converts belongs to the EXL3 pack)
# Bolt:  CONTEXT=131072 (a stream's KV is 8 KiB a token)  THINK_BUDGET=8000 (0: none); no disk states, one stream
# LFM2:  CONTEXT=131072 (KV 20 KiB a token)  THINK_BUDGET=8000 (0: none); one stream, no drafts, no disk states.
#        LFM Open License v1.0: commercial use is licensed only under US$10M annual revenue
# Super: CONTEXT=131072  THINK_BUDGET=8000 (0: none); --precision checkpoint, --vision, MTP drafts; no Clef, no disk states
#        NATIVE=$MODELS/tensorfold-1.1.0-linux-aarch64: the native server's bin/ and lib/ (./prepare.sh nemotron-3.5-super)
set -euo pipefail
shopt -s inherit_errexit            # a failure inside $(image) stops the caller too

MODELS="${MODELS:-$HOME/models}"
BASE="${TF_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
PORT="${PORT:-30000}"
CLEF_PORT="${CLEF_PORT:-30001}"
D1_PORT="${D1_PORT:-30003}"
SRC="$(cd "$(dirname "$0")" && pwd)"
NAMES="tensorfold-serve clef-serve tensorfold-glm tensorfold-bolt tensorfold-lfm2 tensorfold-super d1-serve tensorfold-build"
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/tensorfold-run"

die() { echo "run: $*" >&2; exit 1; }
up() { docker ps --format '{{.Names}}' | grep -qx "$1"; }
avail() { awk '/MemAvailable/{printf "%d", $2 / 1048576}' /proc/meminfo; }
avail_mib() { awk '/MemAvailable/{printf "%d", $2 / 1024}' /proc/meminfo; }

uncache() {  # drop the served checkpoints' files from the page cache (they stay on disk)
    local f
    for f in "$MODELS"/{Qwen3.8-Flash-Next-exl3-3.05bpw,Qwen3.8-Flash-Next-FP4-Local,clef-flash,glm53-flash-mosaic-12l,glm53-flash-sushi-experts-mosaic-dense,GLM-5.3-Flash-DFlash*,Nemotron-3.5-Bolt-74B-A9B-24GB*,LFM2.5-24B-A2B-FP8,NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4,d1-3B}/*.safetensors \
             "$MODELS"/Qwen3.8-Flash-Next-FP4-Local/tensors/*.safetensors \
             "${QWEN_MODEL:-}"/*.safetensors "${QWEN_FP4_MODEL:-}"/*.safetensors "${CLEF_MODEL:-}"/*.safetensors \
             "${GLM_MODEL:-}"/*.safetensors "${BOLT_MODEL:-}"/*.safetensors "${LFM2_MODEL:-}"/*.safetensors \
             "${SUPER_MODEL:-}"/*.safetensors "${D1_MODEL:-}"/*.safetensors; do
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
    tag="tensorfold-run:$(cd "$SRC" && find src pyproject.toml run-constraints.txt -type f -not -name '*.pyc' -print0 | sort -z | xargs -0 sha1sum | sha1sum | cut -c1-12)"
    if ! docker image inspect "$tag" >/dev/null 2>&1; then
        echo "run: installing this checkout into the image $tag (once per source change)" >&2
        docker rm -f tensorfold-build >/dev/null 2>&1 || true
        # a failed install must not be committed under the source's hash, nor prune the images that still work
        docker run --name tensorfold-build -v "$SRC:/tf:ro" "$BASE" \
            bash -c "cp -r /tf /tmp/tf && pip install -q -c /tmp/tf/run-constraints.txt '/tmp/tf[vision]' && rm -rf /tmp/tf" >/dev/null \
            || { docker rm -f tensorfold-build >/dev/null 2>&1; die "installing the checkout into $tag failed"; }
        docker commit --change "LABEL tensorfold.src=$SRC" tensorfold-build "$tag" >/dev/null || die "committing $tag failed"
        docker rm tensorfold-build >/dev/null
        docker image inspect "$tag" >/dev/null 2>&1 || die "$tag is missing after its build"
        # only this checkout's older images: another worktree's (or an unlabelled one a test names) stays
        for old in $(docker images --filter "label=tensorfold.src=$SRC" --format '{{.Repository}}:{{.Tag}}' tensorfold-run); do
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

serve_native() {  # container name, docker flags..., --, tensorfold-native serve arguments
    local name=$1; shift
    local flags=()
    while [ "$1" != "--" ]; do flags+=("$1"); shift; done
    shift
    # upstream's binary from $NATIVE on the base image (any glibc >= 2.28 works); it finds libtfgrammar.so and
    # libturbojpeg.so.0 in the lib/ beside its bin/, and reads libcuda.so.1 from the driver that --gpus mounts
    docker run -d --init --name "$name" --gpus all --ipc=host --network host \
        -v "$MODELS:$MODELS:ro" -v "$NATIVE:/opt/tensorfold:ro" -e HF_HUB_OFFLINE=1 "${flags[@]}" \
        --entrypoint /opt/tensorfold/bin/tensorfold-native "$BASE" serve "$@" --no-update-check >/dev/null
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
    # a swap may hold the loaded model (SWAP_HOLD), drain (SWAP_DRAIN), persist (SWAP_PERSIST), then load: an image build
    # and up to four GLM tries of 15 minutes each
    local wait
    wait=$(awk -v h="${SWAP_HOLD:-0}" -v d="${SWAP_DRAIN:-600}" -v p="${SWAP_PERSIST:-120}" 'BEGIN { printf "%d", h + d + p + 3900 }')
    curl -sf -m "$wait" -X POST "http://127.0.0.1:$PORT/swap/$1" || die "the swap endpoint could not load $1; see $STATE/swap.log"
    echo
}

stop_swap() {  # swap.py leads its own process group (setsid): the run.sh of a load in flight goes with it
    local pid i
    pid=$(cat "$STATE/swap.pid")
    kill -- -"$pid" 2>/dev/null || kill "$pid"
    for i in $(seq 20); do [ -e "/proc/$pid" ] || break; sleep 0.5; done
    rm -f "$STATE/swap.pid"
    echo "run: stopped the swap endpoint"
}

swap() {
    command -v python3 >/dev/null || die "swap needs python3 on the host"
    swapping && die "the swap endpoint is already running (pid $(cat "$STATE/swap.pid")); ./run.sh stop ends it"
    stop_all
    mkdir -p "$STATE"
    rm -f "$STATE/swap.pid"
    PORT=$PORT BACKEND_PORT="${BACKEND_PORT:-30002}" SWAP_PIDFILE="$STATE/swap.pid" \
        setsid nohup python3 "$SRC/swap.py" >>"$STATE/swap.log" 2>&1 </dev/null &
    local i
    for i in $(seq 30); do                       # up to 15 s for the pidfile and the port
        swapping && curl -sf -m 1 "http://127.0.0.1:$PORT/swap" >/dev/null 2>&1 && break
        sleep 0.5
    done
    swapping && curl -sf -m 1 "http://127.0.0.1:$PORT/swap" >/dev/null 2>&1 ||
        { tail -3 "$STATE/swap.log" >&2; die "the swap endpoint did not start"; }
    echo "run: swap endpoint on http://127.0.0.1:$PORT/v1 (log: $STATE/swap.log). A request's \"model\" picks one of: $(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import swap; print(", ".join(swap.TARGET))' "$SRC" 2>/dev/null || echo "the names in swap.py")."
    [ -z "${1:-}" ] || ask_swap "$1"
}

# d1-3B beside the loaded model (not GLM): it starts once that model answers, so the model's startup admission never
# counts it, and it admits itself against what the model left. Its startup estimate at D1_CONTEXT=4096 is 7.3 GiB
# (5.8 GiB of bf16 weights with the vision tower, 1.5 GiB of loading buffers), plus its CUDA context. With the default
# reserve (a tenth of the Spark's memory, 12.2 GiB) it would refuse to start beside most models, hence
# TENSORFOLD_MEMORY_RESERVE_GIB=2. A missing checkpoint or a failed load leaves the model serving without it.
d1() {
    [ "${D1:-0}" != 0 ] || return 0                    # opt-in until its GPU parity and latency are measured
    local model="${D1_MODEL:-$MODELS/d1-3B}"
    if [ ! -e "$model/config.json" ]; then
        echo "run: no d1-3B at $model, so no decision sidecar (./prepare.sh d1, or leave D1 unset)" >&2
        return 0
    fi
    serve d1-serve -e TENSORFOLD_MEMORY_RESERVE_GIB=2 -- "$model" --host 127.0.0.1 --port "$D1_PORT" --name d1-3b \
        --context "${D1_CONTEXT:-4096}"
    # under the swap endpoint run.sh returns once the model answers: requests wait for that, not for d1
    [ -z "${TF_SWAP_CHILD:-}" ] || { echo "run: d1-serve is loading on 127.0.0.1:$D1_PORT; docker logs -f d1-serve"; return 0; }
    wait_ready d1-serve "$D1_PORT" && return 0
    docker logs d1-serve 2>&1 | tail -5 >&2
    docker rm -f d1-serve >/dev/null 2>&1 || true
    echo "run: d1-serve exited while loading; the model serves without it" >&2
}

qwen() {  # [served name] [clef_first: 1 starts Clef before Qwen]
    local model="${QWEN_MODEL:-$MODELS/Qwen3.8-Flash-Next-exl3-3.05bpw}" clef="${CLEF_MODEL:-$MODELS/clef-flash}"
    local name="${1:-qwen3.8-flash-next}" clef_first="${2:-0}"
    local tower="$MODELS/Qwen3.8-Flash-Next-exl3-3.05bpw-vision/vision-f16-v3.safetensors"
    [ -e "$model/config.json" ] || die "no checkpoint at $model (run ./prepare.sh qwen-flash-next)"
    # checked before anything stops: a missing guard must leave the running model loaded
    [ "${GUARD:-1}" = 0 ] || [ -e "$clef/joint_head.safetensors" ] \
        || die "no Clef-Flash at $clef (run ./prepare.sh qwen-flash-next, or GUARD=0)"
    IMAGE=$(image)
    stop_all
    local guard=() budget=() vision=() flags=()
    [ "${GUARD:-1}" = 0 ] || guard=(--thinking-guard "http://127.0.0.1:$CLEF_PORT/v1")
    [ "${THINK_BUDGET:-8000}" = 0 ] || budget=(--thinking-budget "${THINK_BUDGET:-8000}")
    if [ "${VISION:-1}" != 0 ] && [ -e "$tower" ]; then
        vision=(--vision); flags=(-e "TENSORFOLD_VISION_WEIGHTS=$tower")
    fi
    # kept prompt states also go to disk, so a conversation's next turn after a swap or restart reads its state back;
    # Qwen's folder sits beside GLM's checkpoint folders, each holding a subfolder per checkpoint and build
    # FP4-Local's weights leave Clef no room behind Qwen's reserve ("cannot fit requested context 16384"), so Clef starts
    # first there and Qwen's admission sizes its caches around it: 10.1 GiB for caches, 21 GiB left (2026-10-17)
    if [ "${GUARD:-1}" != 0 ] && [ "$clef_first" = 1 ]; then
        serve clef-serve -- "$clef" --host 127.0.0.1 --port "$CLEF_PORT" --name clef --context 16384
        wait_ready clef-serve "$CLEF_PORT" || failed clef-serve
    fi
    [ "${STATES:-1}" = 0 ] || flags+=(-v tensorfold-states:/states -e TENSORFOLD_QWEN_STATES=/states/qwen -e "TENSORFOLD_QWEN_STATES_GIB=${STATES_GIB:-64}" -e "TENSORFOLD_QWEN_STATES_TTL_H=${STATES_TTL_H:-72}" -e "TENSORFOLD_QWEN_STATES_ALL=${STATES_ALL:-0}")
    serve tensorfold-serve "${flags[@]}" -- "$model" --host 127.0.0.1 --port "$PORT" --name "$name" \
        --context "${CONTEXT:-262144}" --max-tokens "${CONTEXT:-262144}" --parallel "${PARALLEL:-4}" --kv-dtype "${KV:-int8}" \
        "${vision[@]}" "${guard[@]}" "${budget[@]}" ${EXTRA:-}
    # Clef starts second: it is ready 17 s before Qwen either way, and Qwen's start is the one a switch waits for
    if [ "${GUARD:-1}" != 0 ] && [ "$clef_first" != 1 ]; then
        serve clef-serve -- "$clef" --host 127.0.0.1 --port "$CLEF_PORT" --name clef --context 16384
        wait_ready clef-serve "$CLEF_PORT" || failed clef-serve
    fi
    wait_ready tensorfold-serve "$PORT" || failed tensorfold-serve
    d1
}

# NVIDIA's QAD FP4-Local checkpoint through the same path: decode 0.71-0.82x the EXL3 pack's, TrueScore 91.0 against
# 87.8 (paired +2.3 [-1.1, +6.4]) and planning +1.5 [-0.4, +3.5], 2026-10-17
qwen_fp4() {
    QWEN_MODEL="${QWEN_FP4_MODEL:-$MODELS/Qwen3.8-Flash-Next-FP4-Local}" VISION="${VISION:-0}" qwen qwen3.8-flash-next-fp4 1
}

bolt() {
    local model="${BOLT_MODEL:-$MODELS/Nemotron-3.5-Bolt-74B-A9B-24GB}" drafter="$MODELS/Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr"
    [ -e "$model/config.json" ] || die "no checkpoint at $model (run ./prepare.sh nemotron-bolt)"
    local draft=() budget=()
    [ -e "$drafter/config.json" ] && draft=(--drafter "$drafter")
    [ "${THINK_BUDGET:-8000}" = 0 ] || budget=(--thinking-budget "${THINK_BUDGET:-8000}")
    IMAGE=$(image)
    stop_all
    serve tensorfold-bolt -- "$model" --host 127.0.0.1 --port "$PORT" --name nemotron-3.5-bolt \
        --context "${CONTEXT:-131072}" --max-tokens "${CONTEXT:-131072}" "${draft[@]}" "${budget[@]}" ${EXTRA:-}
    wait_ready tensorfold-bolt "$PORT" || failed tensorfold-bolt
    d1
}

# Liquid's sampling (generation_config.json: temperature 0.1, top_k 50) with top_p 1.0 where the server's default is
# 0.95; its repetition penalty (1.05) has no keyed rule on the CUDA engines and is not applied
lfm2() {
    local model="${LFM2_MODEL:-$MODELS/LFM2.5-24B-A2B-FP8}"
    [ -e "$model/config.json" ] || die "no checkpoint at $model (run ./prepare.sh lfm2-24b)"
    local budget=()
    [ "${THINK_BUDGET:-8000}" = 0 ] || budget=(--thinking-budget "${THINK_BUDGET:-8000}")
    IMAGE=$(image)
    stop_all
    serve tensorfold-lfm2 -- "$model" --host 127.0.0.1 --port "$PORT" --name lfm2.5-24b-a2b \
        --context "${CONTEXT:-131072}" --max-tokens "${CONTEXT:-131072}" --top-p 1.0 "${budget[@]}" ${EXTRA:-}
    wait_ready tensorfold-lfm2 "$PORT" || failed tensorfold-lfm2
    d1
}

# Upstream TensorFold's native server, not this checkout's Python engine: TrueScore 94.1 and planning 62.6, thinking
# off, ready in 44 s with 40 GiB available after load (2026-10-18). It runs the checkpoint's own FP8/NVFP4 math
# (--precision checkpoint, the Blackwell default), whose replies differ from --precision full's (KL 0.0077 nats).
# No Clef: it guards Qwen. d1 fits beside it: 7.3 GiB of the 40.
super() {
    local model="${SUPER_MODEL:-$MODELS/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4}"
    NATIVE="${NATIVE:-$MODELS/tensorfold-1.1.0-linux-aarch64}"
    [ -e "$model/config.json" ] || die "no checkpoint at $model (run ./prepare.sh nemotron-3.5-super)"
    [ -x "$NATIVE/bin/tensorfold-native" ] || die "no native server at $NATIVE/bin/tensorfold-native (run ./prepare.sh nemotron-3.5-super)"
    [ -e "$NATIVE/lib/libturbojpeg.so.0" ] || echo "run: no $NATIVE/lib/libturbojpeg.so.0, so the server refuses JPEG images (PNG serves)" >&2
    local budget=()
    [ "${THINK_BUDGET:-8000}" = 0 ] || budget=(--thinking-budget "${THINK_BUDGET:-8000}")
    [ "${D1:-0}" = 0 ] || IMAGE=$(image)          # only d1 runs this checkout's image
    stop_all
    serve_native tensorfold-super -- "$model" --host 127.0.0.1 --port "$PORT" --name nemotron-3.5-super \
        --context "${CONTEXT:-131072}" --max-tokens "${CONTEXT:-131072}" --precision checkpoint --vision \
        "${budget[@]}" ${EXTRA:-}
    wait_ready tensorfold-super "$PORT" || failed tensorfold-super
    d1
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
    # Without a budget, single turns of a canon review thought for 17,410 and 21,193 tokens and one ran past 998 s
    # (2026-10-15/16). Each such turn also costs the next one its reasoning again as prompt, at about 310 tok/s.
    local budget=()
    [ "${THINK_BUDGET:-8000}" = 0 ] || budget=(--thinking-budget "${THINK_BUDGET:-8000}")
    # kept prompt states also go to disk, so a conversation's next turn after a swap or restart reads its state back
    local states=()
    [ "${STATES:-1}" = 0 ] || states=(-v tensorfold-states:/states -e TENSORFOLD_GLM_STATES=/states -e "TENSORFOLD_GLM_STATES_GIB=${STATES_GIB:-64}" -e "TENSORFOLD_GLM_STATES_TTL_H=${STATES_TTL_H:-72}")
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
        serve tensorfold-glm -e TF_GLM_DENSE_PROMPT=1 -e TF_GLM_CALIBRATE="${TIMING:-0}" -e TF_GLM_LAZY_SPARSE=1 -e TENSORFOLD_MEMORY_RESERVE_GIB=11 "${states[@]}" -- "$model" --tp 1 --host 127.0.0.1 --port "$PORT" --name glm-5.3-flash \
            --context "${CONTEXT:-$context}" --max-tokens "${CONTEXT:-$context}" "${drafter[@]}" "${budget[@]}" ${EXTRA:-}
        # no d1 beside GLM: GLM leaves 17 GiB, and d1's load took it under 8 GiB (2026-10-17)
        wait_ready tensorfold-glm "$PORT" && return 0
        [[ $(docker logs tensorfold-glm 2>&1) == *'memory budget cannot fit'* ]] || break
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
        [ "$n" = tensorfold-build ] && { echo "tensorfold-build: installing this checkout into an image"; continue; }
        local port=$PORT; [ "$n" = clef-serve ] && port=$CLEF_PORT; [ "$n" = d1-serve ] && port=$D1_PORT
        curl -sf -m 3 "http://127.0.0.1:$port/v1/models" >/dev/null 2>&1 && echo "$n: ready on 127.0.0.1:$port" || echo "$n: loading"
    done
    [ "$any" = 1 ] || echo "nothing is running"
}

command -v docker >/dev/null || die "docker is not installed"
# one start at a time: two direct runs would both stop_all, then load two models side by side. A run that hands the
# load to the swap endpoint (ask_swap) takes no lock: the endpoint's own run.sh child takes it, and would otherwise wait
# behind this process for the whole swap
case "${1:-}" in
    qwen-flash-next|glm-5.3-flash|qwen-flash-next-fp4|nemotron-bolt|lfm2-24b|nemotron-3.5-super)
        if [ -n "${TF_SWAP_CHILD:-}" ] || ! swapping; then mkdir -p "$STATE"; exec 9>"$STATE/run.lock"; flock 9; fi ;;
esac
case "${1:-}" in
    "")              swap qwen-flash-next ;;
    qwen-flash-next) ask_swap "$1" || qwen ;;
    glm-5.3-flash)   ask_swap "$1" || glm ;;
    qwen-flash-next-fp4) ask_swap "$1" || qwen_fp4 ;;
    nemotron-bolt)   ask_swap "$1" || bolt ;;
    lfm2-24b)        ask_swap "$1" || lfm2 ;;
    nemotron-3.5-super) ask_swap "$1" || super ;;
    swap)            swap "${2:-}" ;;
    stop)            if swapping; then stop_swap; fi
                     stop_all ;;
    status)          status ;;
    *) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
