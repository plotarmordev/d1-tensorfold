#!/usr/bin/env bash
# One-time setup, safe to re-run: build the TensorFold image (when its hash tag is missing), download d1-3B at the
# pinned revision and verify every file against checksums/d1-3B.sha256. start.sh runs this first.
#   scripts/prepare.sh            build/download what is missing
#   scripts/prepare.sh --rebuild  rebuild the image even if its tag exists
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
rebuild=0
for arg in "$@"; do
  case "$arg" in --rebuild) rebuild=1 ;; *) die "unknown option $arg" ;; esac
done
validate

stored_hash="$(docker image inspect -f '{{index .Config.Labels "d1.image-hash"}}' "$IMAGE" 2>/dev/null || true)"
if [[ $rebuild == 1 || "$stored_hash" != "$(image_hash)" ]]; then
  log "building $IMAGE (TensorFold ${TF_COMMIT:0:12} + recipe patches)"
  docker build -t "$IMAGE" -f "$RECIPE_DIR/Dockerfile" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg TF_REPO="$TF_REPO" --build-arg TF_COMMIT="$TF_COMMIT" \
    --build-arg IMAGE_EXTRAS="$IMAGE_EXTRAS" --build-arg IMAGE_HASH="$(image_hash)" "$RECIPE_DIR"
else
  log "image $IMAGE is built (use --rebuild to force)"
fi

mkdir -p "$MODEL_DIR" "$CACHE_DIR"
if (cd "$MODEL_DIR" && sha256sum --quiet -c "$RECIPE_DIR/checksums/d1-3B.sha256" >/dev/null 2>&1); then
  log "$MODEL_REPO@${MODEL_REVISION:0:12} is in $MODEL_DIR and verified"
else
  log "downloading $MODEL_REPO@${MODEL_REVISION:0:12} to $MODEL_DIR"
  docker run --rm -u "$(id -u):$(id -g)" -e HF_TOKEN -e HF_HOME=/tmp/hf -v "$MODEL_DIR:/model" "$IMAGE" \
    hf download "$MODEL_REPO" --revision "$MODEL_REVISION" --local-dir /model
  (cd "$MODEL_DIR" && sha256sum --quiet -c "$RECIPE_DIR/checksums/d1-3B.sha256") \
    || die "the downloaded files do not match checksums/d1-3B.sha256"
  log "verified"
fi
