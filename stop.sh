#!/usr/bin/env bash
# Stop the TensorFold server (and the llama.cpp baseline, if it runs).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/config.sh"
for name in "$CONTAINER_NAME" "$LLAMACPP_CONTAINER"; do
  if docker container inspect "$name" >/dev/null 2>&1; then
    docker rm -f "$name" >/dev/null && log "stopped $name"
  fi
done
