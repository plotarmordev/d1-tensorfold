#!/usr/bin/env bash
# Serve d1-3B on TensorFold: POST /v1/systemone (llama-server's System One API) and /v1/decisions.
#   ./start.sh                     the default precision (bf16) on 127.0.0.1:8080
#   PRECISION=fp8 ./start.sh       8-bit weights (see README: Precisions)
# Every setting is checked before anything is stopped, so a typo leaves a running server alone.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/config.sh"
validate
"$RECIPE_DIR/scripts/prepare.sh"
check_memory

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  log "replacing the running $CONTAINER_NAME"
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi
log "starting $SERVED_NAME ($PRECISION) on $HOST:$PORT"
docker run -d --name "$CONTAINER_NAME" --gpus all --ipc=host --ulimit memlock=-1 \
  -u "$(id -u):$(id -g)" -e HOME=/tmp -p "$HOST:$PORT:8080" -v "$MODEL_DIR:/model:ro" -v "$CACHE_DIR:/cache" \
  -e TENSORFOLD_LFM2_PRECISION="$PRECISION" -e TENSORFOLD_LFM2_TOKEN_BUDGET="$TOKEN_BUDGET" \
  -e TENSORFOLD_LFM2_GRAPHS="$CUDA_GRAPHS" \
  "$IMAGE" tensorfold serve /model --backend cuda --no-drafts --host 0.0.0.0 --port 8080 --name "$SERVED_NAME" \
  >/dev/null
# First start compiles kernels for this GPU; later starts reuse the mounted cache.
probe_host="$HOST"
[[ "$probe_host" != 0.0.0.0 ]] || probe_host=127.0.0.1
if ! wait_healthy "http://$probe_host:$PORT/health" 1800 "$CONTAINER_NAME"; then
  docker logs --tail 40 "$CONTAINER_NAME" >&2 || true
  docker rm -f "$CONTAINER_NAME" >/dev/null
  die "$CONTAINER_NAME did not become healthy (startup logs above)"
fi
log "ready: http://$HOST:$PORT/v1/systemone"
