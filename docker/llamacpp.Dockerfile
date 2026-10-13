# The llama.cpp baseline: llama-server at a pinned commit, built with CUDA for one GPU architecture.
# Built by scripts/llamacpp.sh with the pins from scripts/config.sh.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
FROM ${BASE_IMAGE}
ARG LLAMACPP_REPO=https://github.com/ggml-org/llama.cpp.git
ARG LLAMACPP_COMMIT
ARG CUDA_ARCH=121
RUN test -n "${LLAMACPP_COMMIT}" && git init /opt/llama.cpp && cd /opt/llama.cpp && \
    git fetch --depth 1 "${LLAMACPP_REPO}" "${LLAMACPP_COMMIT}" && git checkout FETCH_HEAD && \
    cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" -DLLAMA_CURL=OFF \
          -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build -j"$(nproc)" --target llama-server && \
    ln -s /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
WORKDIR /opt/llama.cpp
