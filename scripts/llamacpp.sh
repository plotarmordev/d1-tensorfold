#!/usr/bin/env bash
# The llama.cpp baseline: llama-server at the pinned commit serving the same model from LiquidAI's GGUF files.
#   scripts/llamacpp.sh start [BF16|Q8_0|Q4_K_M]    build/download what is missing, serve on 127.0.0.1:$LLAMACPP_PORT
#   scripts/llamacpp.sh stop
# The flags are the ones the published numbers were measured with: every layer on the GPU, flash attention,
# 64 slots of 4096 tokens and 4096-token batches (so the 3.4k-token state is one batch and 64 requests run at once).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
cmd="${1:-}"
quant="${2:-BF16}"
validate

case "$cmd" in
  stop)
    docker rm -f "$LLAMACPP_CONTAINER" >/dev/null 2>&1 && log "stopped $LLAMACPP_CONTAINER" || true
    exit 0 ;;
  start) ;;
  *) die "usage: scripts/llamacpp.sh start [BF16|Q8_0|Q4_K_M] | stop" ;;
esac
case "$quant" in BF16|Q8_0|Q4_K_M) ;; *) die "quantization $quant: BF16, Q8_0 or Q4_K_M" ;; esac

if ! docker image inspect "$LLAMACPP_IMAGE" >/dev/null 2>&1; then
  log "building $LLAMACPP_IMAGE (llama.cpp ${LLAMACPP_COMMIT:0:12}, sm_$CUDA_ARCH)"
  docker build -t "$LLAMACPP_IMAGE" -f "$RECIPE_DIR/docker/llamacpp.Dockerfile" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg LLAMACPP_REPO="$LLAMACPP_REPO" \
    --build-arg LLAMACPP_COMMIT="$LLAMACPP_COMMIT" --build-arg CUDA_ARCH="$CUDA_ARCH" "$RECIPE_DIR/docker"
fi

files=("d1-3B-$quant.gguf" "mmproj-d1-3B-BF16.gguf")
mkdir -p "$GGUF_DIR"
verify() { (cd "$GGUF_DIR" && grep -F -e "${files[0]}" -e "${files[1]}" "$RECIPE_DIR/checksums/d1-3B-GGUF.sha256" \
  | sha256sum --quiet -c >/dev/null 2>&1); }
if ! verify; then
  log "downloading ${files[*]} from $GGUF_REPO@${GGUF_REVISION:0:12}"
  docker run --rm -u "$(id -u):$(id -g)" -e HF_TOKEN -e HF_HOME=/tmp/hf -v "$GGUF_DIR:/gguf" "$LLAMACPP_IMAGE" \
    hf download "$GGUF_REPO" "${files[@]}" --revision "$GGUF_REVISION" --local-dir /gguf
  verify || die "the GGUF files do not match checksums/d1-3B-GGUF.sha256"
fi

check_memory
docker rm -f "$LLAMACPP_CONTAINER" >/dev/null 2>&1 || true
log "starting llama-server ($quant) on 127.0.0.1:$LLAMACPP_PORT"
docker run -d --name "$LLAMACPP_CONTAINER" --gpus all --ipc=host -p "127.0.0.1:$LLAMACPP_PORT:8080" \
  -v "$GGUF_DIR:/gguf:ro" "$LLAMACPP_IMAGE" \
  llama-server -m "/gguf/${files[0]}" --mmproj "/gguf/${files[1]}" -ngl 99 -fa on -c 262144 -np 64 \
  -b 4096 -ub 4096 --host 0.0.0.0 --port 8080 >/dev/null
if ! wait_healthy "http://127.0.0.1:$LLAMACPP_PORT/health" 600 "$LLAMACPP_CONTAINER"; then
  docker logs --tail 40 "$LLAMACPP_CONTAINER" >&2 || true
  docker rm -f "$LLAMACPP_CONTAINER" >/dev/null
  die "llama-server did not become healthy (startup logs above)"
fi
log "ready: http://127.0.0.1:$LLAMACPP_PORT/v1/systemone"
