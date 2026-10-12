#!/usr/bin/env bash
# Download what run.sh serves on one DGX Spark, into $MODELS (default ~/models), and build the image-input weights.
#
#   ./prepare.sh                  both models
#   ./prepare.sh qwen-flash-next  Qwen3.8 Flash Next (80 GB), Clef-Flash (18 GB), image tower (0.9 GB)
#   ./prepare.sh glm-5.3-flash    GLM-5.3 Flash Mosaic 12L (90 GB) and its draft model (2.3 GB)
#
# Needs Docker with the NVIDIA runtime. Downloads run inside the serving image, so the host needs no Python packages.
# A gated or rate-limited download reads HF_TOKEN from the environment. Reruns resume and skip what is complete.
#
# GLM_DRAFTER_REPO picks the GLM draft model. The default, cfontes/GLM-5.3-Flash-DFlash2-TR3-v3, is a finetune of
# incoai/GLM-5.3-Flash-DFlash2 (CC-BY-NC-ND-4.0): personal use. modal-labs/GLM-5.3-Flash-DFlash is MIT and 3-4% slower.
set -euo pipefail

MODELS="${MODELS:-$HOME/models}"
IMAGE="${TF_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
GLM_DRAFTER_REPO="${GLM_DRAFTER_REPO:-cfontes/GLM-5.3-Flash-DFlash2-TR3-v3}"
SRC="$(cd "$(dirname "$0")" && pwd)"

# repo, revision, folder under $MODELS: the revisions the fork's numbers were measured on
QWEN=(turboderp/Qwen3.8-Flash-Next-exl3 69e33439ae950f17bcbe95c98f117d80f759ab6d Qwen3.8-Flash-Next-exl3-3.05bpw)
CLEF=(Cloudflare/clef-flash 17f0b0ad64efb65d273590632833508766b2aae6 clef-flash)
GLM=(0xSero/GLM-5.3-Flash-EXL3-Spark 2642851741fc833764e77d03039117be559dc83e glm53-flash-mosaic-12l)

die() { echo "prepare: $*" >&2; exit 1; }

fetch() {  # repo revision folder
    local repo=$1 rev=$2 dir="$MODELS/$3"
    echo "prepare: $repo -> $dir"
    docker run --rm --network host --user "$(id -u):$(id -g)" -e HOME=/tmp -e HF_TOKEN="${HF_TOKEN:-}" \
        -v "$MODELS:$MODELS" "$IMAGE" \
        hf download "$repo" ${rev:+--revision "$rev"} --local-dir "$dir" >/dev/null
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

glm() {
    fetch "${GLM[@]}"
    fetch "$GLM_DRAFTER_REPO" "" "$(basename "$GLM_DRAFTER_REPO")"
}

command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "cannot reach the Docker daemon (is this user in the docker group?)"
docker info 2>/dev/null | grep -qi nvidia || die "Docker has no NVIDIA runtime (install the NVIDIA Container Toolkit)"
mkdir -p "$MODELS"
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"

what="${1:-all}"
need=0
case "$what" in
    qwen-flash-next) need=100 ;;
    glm-5.3-flash)   need=93 ;;
    all)             need=193 ;;
    *) die "unknown target '$what' (qwen-flash-next, glm-5.3-flash, or nothing for both)" ;;
esac
free=$(df -BG --output=avail "$MODELS" | tail -1 | tr -dc 0-9)
[ "$free" -ge "$need" ] || echo "prepare: $MODELS has $free GB free and a first download needs $need GB" >&2

case "$what" in
    qwen-flash-next) qwen ;;
    glm-5.3-flash)   glm ;;
    all)             qwen; glm ;;
esac
echo "prepare: done. Start a model with ./run.sh qwen-flash-next or ./run.sh glm-5.3-flash"
