#!/usr/bin/env bash
# Download what run.sh serves on one DGX Spark, into $MODELS (default ~/models), and build the image-input weights.
#
#   ./prepare.sh                  both models
#   ./prepare.sh qwen-flash-next  Qwen3.8 Flash Next (80 GB), Clef-Flash (18 GB), image tower (0.9 GB)
#   ./prepare.sh glm-5.3-flash    GLM-5.3 Flash hybrid (Sushi experts on Mosaic 12L's dense part, 182 GB of downloads)
#                                 and its draft model (2.3 GB)
#   ./prepare.sh qwen-flash-next-fp4   NVIDIA's Qwen3.8 Flash Next FP4-Local (96 GB) and Clef-Flash
#   ./prepare.sh nemotron-bolt    Nemotron 3.5 Bolt 74B-A9B 24 GB (23 GB) and its LiLiCorr draft model (0.9 GB)
#
# Needs Docker with the NVIDIA runtime. Downloads run inside the serving image, so the host needs no Python packages.
# A gated or rate-limited download reads HF_TOKEN from the environment. Reruns resume and skip what is complete.
#
# GLM_DRAFTER_REPO picks the GLM draft model. The default, cfontes/GLM-5.3-Flash-DFlash2-TR3-v3, is a finetune of
# incoai/GLM-5.3-Flash-DFlash2 (CC-BY-NC-ND-4.0): personal use. modal-labs/GLM-5.3-Flash-DFlash is MIT and 3-4% slower.
# Both download at a pinned revision; another repo needs GLM_DRAFTER_REV.
#
# GLM_PACK=mosaic fetches Mosaic 12L alone (90 GB). The default also fetches the Sushi pack's 42 routed-expert files
# (82 GB) and assembles the hybrid run.sh serves: TrueScore 93.6 against Mosaic's 86.5, 5-7% slower (2026-10-13).
set -euo pipefail

MODELS="${MODELS:-$HOME/models}"
IMAGE="${TF_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
GLM_DRAFTER_REPO="${GLM_DRAFTER_REPO:-cfontes/GLM-5.3-Flash-DFlash2-TR3-v3}"
declare -A DRAFTER_REV=(
    [cfontes/GLM-5.3-Flash-DFlash2-TR3-v3]=45f7e139584e299f9850c9f92df2c4563a470432
    [modal-labs/GLM-5.3-Flash-DFlash]=dae6d319510889a7f1ac5242d5b7fb2d5d95eb05
)
GLM_DRAFTER_REV="${GLM_DRAFTER_REV:-${DRAFTER_REV[$GLM_DRAFTER_REPO]:-}}"
SRC="$(cd "$(dirname "$0")" && pwd)"

# repo, revision, folder under $MODELS: the revisions the fork's numbers were measured on
QWEN=(turboderp/Qwen3.8-Flash-Next-exl3 69e33439ae950f17bcbe95c98f117d80f759ab6d Qwen3.8-Flash-Next-exl3-3.05bpw)
CLEF=(Cloudflare/clef-flash 17f0b0ad64efb65d273590632833508766b2aae6 clef-flash)
GLM=(0xSero/GLM-5.3-Flash-EXL3-Spark 2642851741fc833764e77d03039117be559dc83e glm53-flash-mosaic-12l)
SUSHI=(beamster/GLM-5.3-Flash-Sushi-2.4bpw 0bc3bf157723d2bb7723e31118addf48b22101b0 glm53-flash-sushi-2.4bpw)
HYBRID=glm53-flash-sushi-experts-mosaic-dense
QWEN_FP4=(nvidia/Qwen3.8-Flash-Next-FP4-Local b9ce18a6887e18c089fdea8509a33e63429a3300 Qwen3.8-Flash-Next-FP4-Local)
BOLT=(nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB c6cc2e8ffa353403213d6b9b18aaf8eb684d705a Nemotron-3.5-Bolt-74B-A9B-24GB)
BOLT_DRAFT=(nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr abd478065678ee83a93821c4b3f512da24e4cecb Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr)

die() { echo "prepare: $*" >&2; exit 1; }

fetch() {  # repo revision folder [file ...]: the whole repo, or the files named
    local repo=$1 rev=$2 dir="$MODELS/$3"
    shift 3
    echo "prepare: $repo -> $dir"
    # -e HF_TOKEN with no value copies it from this environment, so the token stays out of the command line
    docker run --rm --network host --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_TOKEN \
        -v "$MODELS:$MODELS" "$IMAGE" \
        hf download "$repo" "$@" ${rev:+--revision "$rev"} --local-dir "$dir" >/dev/null
    [ -e "$dir/config.json" ] || die "$dir has no config.json after the download"
}

qwen() {
    fetch "${QWEN[@]}"
    fetch "${CLEF[@]}"
    # The pack stores its image tower quantized inside the shards; serving reads an FP16 copy, converted once on the CPU.
    local out="$MODELS/${QWEN[2]}-vision/vision-f16-v3.safetensors"
    if [ -e "$out" ]; then
        echo "prepare: image tower already at $out"
        return
    fi
    echo "prepare: converting the image tower (about 2 minutes)"
    mkdir -p "$(dirname "$out")"
    docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$MODELS:$MODELS" -v "$SRC:/tf:ro" "$IMAGE" \
        bash -c "cp -r /tf /tmp/tf && pip install -q --user --no-warn-script-location '/tmp/tf[vision]' >/dev/null && \
                 python -m tensorfold.vision.exl3_convert '$MODELS/${QWEN[2]}' '$out'"
}

qwen_fp4() {
    fetch "${QWEN_FP4[@]}"
    fetch "${CLEF[@]}"
}

bolt() {
    fetch "${BOLT[@]}"
    fetch "${BOLT_DRAFT[@]}"
}

glm() {
    fetch "${GLM[@]}"
    [ -n "$GLM_DRAFTER_REV" ] || die "no pinned revision for $GLM_DRAFTER_REPO: set GLM_DRAFTER_REV to a commit"
    fetch "$GLM_DRAFTER_REPO" "$GLM_DRAFTER_REV" "$(basename "$GLM_DRAFTER_REPO")"
    [ "${GLM_PACK:-hybrid}" = mosaic ] && return
    # Only the routed experts of MoE layers 3-44: the pack's dense part, MTP layer (L45) and vision tower go unused
    local i experts=(config.json)
    for i in $(seq -w 3 44); do experts+=("model-exl3-L$i.safetensors"); done
    fetch "${SUSHI[@]}" "${experts[@]}"
    if [ -e "$MODELS/$HYBRID/config.json" ]; then
        echo "prepare: hybrid already at $MODELS/$HYBRID"
        return
    fi
    # Copies Mosaic's dense tensors (6.1 GB) and links the expert files, so the Sushi folder must stay
    echo "prepare: assembling $MODELS/$HYBRID"
    docker run --rm --user "$(id -u):$(id -g)" -v "$MODELS:$MODELS" -v "$SRC:/tf:ro" "$IMAGE" \
        python /tf/tools/glm5_stacked_experts_pack.py --dense "$MODELS/${GLM[2]}" --experts "$MODELS/${SUSHI[2]}" \
        --out "$MODELS/$HYBRID"
}

command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "cannot reach the Docker daemon (is this user in the docker group?)"
docker info 2>/dev/null | grep -qi nvidia || die "Docker has no NVIDIA runtime (install the NVIDIA Container Toolkit)"
mkdir -p "$MODELS"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"

what="${1:-all}"
glm_need=$([ "${GLM_PACK:-hybrid}" = mosaic ] && echo 93 || echo 182)   # GB, the draft model included
case "$what" in
    qwen-flash-next) need=100 ;;
    glm-5.3-flash)   need=$glm_need ;;
    qwen-flash-next-fp4) need=115 ;;
    nemotron-bolt)   need=25 ;;
    all)             need=$((100 + glm_need)) ;;
    *) die "unknown target '$what' (qwen-flash-next, glm-5.3-flash, qwen-flash-next-fp4, nemotron-bolt, or nothing for the first two)" ;;
esac
free=$(df -BG --output=avail "$MODELS" | tail -1 | tr -dc 0-9)
[ "$free" -ge "$need" ] || echo "prepare: $MODELS has $free GB free and a first download needs $need GB" >&2

case "$what" in
    qwen-flash-next) qwen ;;
    glm-5.3-flash)   glm ;;
    qwen-flash-next-fp4) qwen_fp4 ;;
    nemotron-bolt)   bolt ;;
    all)             qwen; glm ;;
esac
echo "prepare: done. Start a model with ./run.sh qwen-flash-next or ./run.sh glm-5.3-flash"
