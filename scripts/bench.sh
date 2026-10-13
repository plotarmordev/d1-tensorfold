#!/usr/bin/env bash
# Benchmarks and parity, run from the TensorFold image (the client needs only Python's standard library and the
# model's tokenizer, which the image has), against any /v1/systemone server.
#   scripts/bench.sh run URL LABEL    latency + throughput -> results/LABEL.json, parity set -> results/parity-LABEL.json
#   scripts/bench.sh reference [LABEL]  model's own code -> results/LABEL.json (default reference-local; GPU)
#   scripts/bench.sh report           the tables in README.md, from results/*.json
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "build the image first: scripts/prepare.sh"
mkdir -p "$RECIPE_DIR/results"
in_image() { docker run --rm --network host -u "$(id -u):$(id -g)" -e HOME=/tmp -v "$RECIPE_DIR:/recipe" \
  -v "$MODEL_DIR:/model:ro" "$@"; }

case "${1:-}" in
  run)
    url="${2:?usage: scripts/bench.sh run URL LABEL}"; label="${3:?usage: scripts/bench.sh run URL LABEL}"
    [[ "$label" =~ ^[A-Za-z0-9._-]+$ ]] || die "LABEL: letters, digits, dot, dash, underscore"
    [[ ! -e "$RECIPE_DIR/results/$label.json" && ! -e "$RECIPE_DIR/results/parity-$label.json" ]] \
      || die "results for $label already exist; use a fresh LABEL"
    in_image "$IMAGE" bash -c '
      python bench/systemone_bench.py --url "$1" --tokenizer /model/tokenizer.json \
        --out "results/$2.json" --label "$2" &&
      python bench/parity.py --url "$1" --dump "results/parity-$2.json" &&
      python bench/parity.py --compare results/reference.json "results/parity-$2.json"
    ' bash "$url" "$label" ;;
  reference)
    reference_label="${2:-reference-local}"
    [[ "$reference_label" =~ ^[A-Za-z0-9._-]+$ ]] || die "invalid reference label"
    [[ ! -e "$RECIPE_DIR/results/$reference_label.json" ]] || die "reference output already exists; use a fresh label"
    validate
    check_memory
    mkdir -p "$CACHE_DIR"
    in_image --gpus all --ipc=host -v "$CACHE_DIR:/cache" -e HF_HOME=/cache/huggingface "$IMAGE" \
      python reference/transformers_ref.py --model /model --tokenizer /model/tokenizer.json \
      --out "results/$reference_label.json" ;;
  report)
    in_image "$IMAGE" python bench/report.py results ;;
  *) die "usage: scripts/bench.sh run URL LABEL | reference | report" ;;
esac
