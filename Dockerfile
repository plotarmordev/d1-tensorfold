# d1-3B on TensorFold (CUDA, Python engine line): TensorFold at a pinned commit plus this recipe's patches.
# Built by scripts/prepare.sh, which passes the pins from scripts/config.sh. TensorFold compiles its CUDA
# extensions on first start for the GPU it finds (into /cache, a volume), so the image is not tied to one GPU.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c
FROM ${BASE_IMAGE}
ARG TF_REPO=https://github.com/ashhart/TensorFold.git
ARG TF_COMMIT
ARG IMAGE_EXTRAS
RUN test -n "${TF_COMMIT}" && \
    pip install --no-cache-dir "tensorfold @ git+${TF_REPO}@${TF_COMMIT}" ${IMAGE_EXTRAS} && tensorfold --version
COPY patches /opt/d1-patches
COPY checksums/tensorfold-audited.sha256 /opt/d1-source.sha256
RUN cd "$(python -c 'import os, tensorfold; print(os.path.dirname(os.path.dirname(tensorfold.__file__)))')" && \
    for p in /opt/d1-patches/*.patch; do echo "applying $p"; patch -p1 --forward < "$p" || exit 1; done && \
    sha256sum --strict -c /opt/d1-source.sha256 && \
    python -c "import tensorfold.families.lfm2.cuda.engine, tensorfold.families.lfm2.cuda.app, tensorfold.cuda.server"
ARG IMAGE_HASH
LABEL d1.image-hash=${IMAGE_HASH}
ENV TORCH_EXTENSIONS_DIR=/cache/torch_extensions TRITON_CACHE_DIR=/cache/triton HF_HOME=/cache/huggingface
WORKDIR /recipe
