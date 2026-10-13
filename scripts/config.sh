# Sourced by the launchers. Precedence: caller settings > scripts/local.sh > defaults.
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Preserve caller settings while loading the operator's optional, git-ignored file.
declare -A _caller=()
for _key in MODEL_REPO MODEL_REVISION GGUF_REPO GGUF_REVISION TF_REPO TF_COMMIT LLAMACPP_REPO LLAMACPP_COMMIT \
  BASE_IMAGE CUDA_ARCH IMAGE_EXTRAS DATA_DIR MODEL_DIR GGUF_DIR CACHE_DIR PRECISION HOST PORT SERVED_NAME \
  CONTEXT TOKEN_BUDGET CUDA_GRAPHS CONTAINER_NAME LLAMACPP_PORT LLAMACPP_CONTAINER LLAMACPP_IMAGE IMAGE MIN_FREE_GIB; do
  if [[ -v $_key ]]; then _caller[$_key]="${!_key}"; fi
done
[[ ! -f "$RECIPE_DIR/scripts/local.sh" ]] || source "$RECIPE_DIR/scripts/local.sh"
for _key in "${!_caller[@]}"; do printf -v "$_key" '%s' "${_caller[$_key]}"; done
unset _caller _key

MODEL_REPO="${MODEL_REPO:-LiquidAI/d1-3B}"
MODEL_REVISION="${MODEL_REVISION:-051bcc464b01b9f92942b364d9586b0ef5912432}"
GGUF_REPO="${GGUF_REPO:-LiquidAI/d1-3B-GGUF}"
GGUF_REVISION="${GGUF_REVISION:-bb1e436ea78eb96a3f1acb6da865f70c2fbeb563}"
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_COMMIT="${TF_COMMIT:-ed78d6fc204d89d90b045bf033d6551e7714f3a1}" # Python 0.6 line
LLAMACPP_REPO="${LLAMACPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
LLAMACPP_COMMIT="${LLAMACPP_COMMIT:-4413a192410fad1f054e9216114087a3e01527a0}"
# NGC PyTorch 26.07, immutable manifest (includes arm64).
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c}"
CUDA_ARCH="${CUDA_ARCH:-121}"
IMAGE_EXTRAS="${IMAGE_EXTRAS:-transformers==5.19.0 huggingface-hub==1.33.0 tokenizers==0.23.1 safetensors==0.8.0 pillow==12.3.0 jinja2==3.1.6 numpy==2.1.0}"

DATA_DIR="$(realpath -m "${DATA_DIR:-$RECIPE_DIR/data}")"
MODEL_DIR="$(realpath -m "${MODEL_DIR:-$DATA_DIR/models/d1-3B}")"
GGUF_DIR="$(realpath -m "${GGUF_DIR:-$DATA_DIR/models/d1-3B-GGUF}")"
CACHE_DIR="$(realpath -m "${CACHE_DIR:-$DATA_DIR/cache}")"
PRECISION="${PRECISION:-bf16}" # native checkpoint BF16 only
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
SERVED_NAME="${SERVED_NAME:-d1-3B}"
CONTEXT="${CONTEXT:-32768}"
TOKEN_BUDGET="${TOKEN_BUDGET:-65536}"
CUDA_GRAPHS="${CUDA_GRAPHS:-1}"
MIN_FREE_GIB="${MIN_FREE_GIB:-56}" # launch floor for shared CPU/GPU memory; not a runtime allocation cap
CONTAINER_NAME="${CONTAINER_NAME:-d1-tensorfold}"
LLAMACPP_PORT="${LLAMACPP_PORT:-8081}"
LLAMACPP_CONTAINER="${LLAMACPP_CONTAINER:-d1-llamacpp}"

image_hash() {
  { printf '%s\n' "$TF_REPO" "$TF_COMMIT" "$BASE_IMAGE" "$IMAGE_EXTRAS";
    cat "$RECIPE_DIR/Dockerfile" "$RECIPE_DIR"/patches/*.patch "$RECIPE_DIR/checksums/tensorfold-audited.sha256";
  } | sha256sum | cut -c1-12
}
llamacpp_hash() {
  { printf '%s\n' "$LLAMACPP_REPO" "$LLAMACPP_COMMIT" "$BASE_IMAGE" "$CUDA_ARCH";
    cat "$RECIPE_DIR/docker/llamacpp.Dockerfile"; } | sha256sum | cut -c1-12
}
IMAGE="${IMAGE:-d1-tensorfold:$(image_hash)}"
LLAMACPP_IMAGE="${LLAMACPP_IMAGE:-d1-llamacpp:$(llamacpp_hash)}"
log() { printf '[d1-tensorfold] %s\n' "$*" >&2; }
die() { log "error: $*"; exit 1; }
validate() {
  [[ "$PRECISION" == bf16 ]] || die "PRECISION: this audited recipe supports native bf16 only"
  for value in "$PORT" "$LLAMACPP_PORT"; do
    [[ "$value" =~ ^[1-9][0-9]{0,4}$ && "$value" -le 65535 ]] || die "invalid port: $value"
  done
  [[ "$CONTEXT" =~ ^[1-9][0-9]{0,4}$ && "$CONTEXT" -le 32768 ]] || die "CONTEXT: 1 through 32768"
  [[ "$TOKEN_BUDGET" =~ ^[1-9][0-9]{0,6}$ && "$TOKEN_BUDGET" -ge 4096 && "$TOKEN_BUDGET" -le 65536 ]] \
    || die "TOKEN_BUDGET: 4096 through 65536"
  [[ "$MIN_FREE_GIB" =~ ^[1-9][0-9]{0,3}$ ]] || die "MIN_FREE_GIB must be positive"
  [[ "$CUDA_GRAPHS" == 0 || "$CUDA_GRAPHS" == 1 ]] || die "CUDA_GRAPHS: 0 or 1"
  command -v docker >/dev/null || die "docker is required"
}
check_memory() {
  local key value unit available=0
  while read -r key value unit; do
    if [[ "$key" == MemAvailable: ]]; then available=$value; break; fi
  done < /proc/meminfo
  (( available >= MIN_FREE_GIB * 1024 * 1024 )) \
    || die "need $MIN_FREE_GIB GiB MemAvailable before loading; found $((available / 1024 / 1024)) GiB"
}
wait_healthy() {
  local url="$1" limit="${2:-900}" name="${3:-}" waited=0
  until curl --noproxy '*' --max-time 3 -sf "$url" >/dev/null 2>&1; do
    if [[ -n "$name" ]] && [[ "$(docker inspect -f '{{.State.Running}}' "$name")" != true ]]; then return 1; fi
    sleep 2; waited=$((waited + 5))
    [[ $waited -lt $limit ]] || return 1
  done
}
