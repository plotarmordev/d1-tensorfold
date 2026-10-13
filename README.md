# d1-3B on TensorFold — DGX Spark recipe

Serve [Liquid AI's d1-3B](https://huggingface.co/LiquidAI/d1-3B) on one NVIDIA DGX Spark using
[TensorFold's Python 0.6 engine](https://github.com/ashhart/TensorFold/tree/python-0.6).
This repository supplies the missing LFM2 CUDA family as patches, a reproducible container build,
launchers, and a head-to-head benchmark against [llama.cpp](https://github.com/ggml-org/llama.cpp).
It does not use TensorFold's separate native 1.x engine.

**Use BF16 first.** FP8 trades some probability fidelity for lower short-request latency. INT4 is
an additional lossy option, not a blanket recommendation. See the measured [latency, throughput,
CV and parity tables](results/report.md) and the [raw samples](results/).

A d1 decision is a forward pass, not generated text. The server reads logits for the question's
option tokens and returns probabilities with **zero output tokens**. Chat/completion generation
is deliberately refused.

## Quick start

Requirements: Linux, Docker with NVIDIA Container Toolkit, a CUDA-13-compatible NVIDIA driver,
Bash, curl, and GNU coreutils. Tested hardware: **one GB10 / DGX Spark, arm64, sm_121**.
Other GPUs and multi-GPU operation are not qualified by this recipe.
Allow roughly 35 GB for the image and d1 checkpoint; the optional three GGUF baselines use another 11 GB.

```bash
git clone https://github.com/plotarmordev/d1-tensorfold.git
cd d1-tensorfold
./scripts/prepare.sh
./start.sh
```

Repository access is required while this project is private. `prepare.sh` builds the image, downloads the
pinned checkpoint, and verifies its files with SHA-256. It is safe to run again. `start.sh` calls it first.
First startup compiles CUDA/Triton kernels for the visible GPU into `data/cache`; later launches reuse them.
Compilation and model loading are excluded from the warm-request measurements.

The default endpoint is `http://127.0.0.1:8080/v1/systemone`:

```bash
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice this month, please refund one of them.",
       "questions":{"refund":{"type":"noul","instructions":"Is the customer asking for a refund?"}}}'
```

`questions` is an **object keyed by question ID**, not a list. Supported types:

| Type | Criteria | Result |
|---|---|---|
| `noul` | Optional true/false descriptions | `noul`: probability of yes |
| `choice` | Object mapping option names to descriptions | Chosen name, confidence, probabilities |
| `score` | List of 2–10 ordered descriptions | Expected numeric level, confidence, probabilities, legend |

For images, provide `images: ["data:image/jpeg;base64,..."]` and either textual `state` or `state: null`.
The engine uses the checkpoint's SigLIP2 tower and projector, not an external vision model. It accepts up to
eight images and caps each at one megapixel before processing. The benchmark covers **one** 384×288 image;
it does not establish quality parity for arbitrary resizing, multi-image requests, or video.

TensorFold's separate `/v1/decisions` schema also works; it is not an alias for System One. That schema
requires full-vocabulary probability mass, so its adapter computes the full head. The optimized System One
path does not. `/health` and `/v1/models` are available.

```bash
PRECISION=fp8 ./start.sh       # replace this recipe's TensorFold container
./stop.sh                    # stop this recipe's TensorFold and baseline containers
```

### Settings and memory

Settings live in [scripts/config.sh](scripts/config.sh). Copy `scripts/local.sh.example` to
`scripts/local.sh` for persistent local settings. Caller/environment values override that file;
that file overrides defaults. Local settings, models, caches and downloaded pictures are git-ignored.

| Setting | Default | Meaning |
|---|---|---|
| `PRECISION` | `bf16` | `bf16`, `fp8`, or `int4` |
| `HOST`, `PORT` | `127.0.0.1`, `8080` | TensorFold port binding |
| `LLAMACPP_PORT` | `8081` | Loopback-only baseline port; override if occupied |
| `DATA_DIR` | `./data` | Model downloads and caches |
| `TOKEN_BUDGET` | `65536` | Total tokens in one packed forward, allowed range 4096–65536 |
| `CUDA_GRAPHS` | `1` | Set `0` to disable short-request CUDA graphs |
| `MIN_FREE_GIB` | `56` | Host MemAvailable required before a model launch |
| `CUDA_ARCH` | `121` | llama.cpp build architecture |

GB10 shares CPU and GPU memory. The launch floor is deliberately conservative: it leaves headroom for
weight conversion, BF16 copies, graph pools, vision activations and other processes. Requests are limited
by the model's 32768-token context and the packed-forward budget; over-limit requests return an error.
The launch floor is **not** a runtime memory cap or watchdog. Reserve the device operationally, do not
launch overlapping benchmarks, and do not lower the floor without measuring your own peak usage.
Containers do not auto-restart, which would bypass the launch memory check.
One validation launch failed to create a CUDA context even with sufficient `MemAvailable`; much of that
memory was reclaimable file cache rather than immediately free pages. This check cannot guarantee a
successful CUDA allocation. Do not treat a context-allocation failure as permission to reset the GPU or
stop unrelated services.

Binding `HOST=0.0.0.0` exposes an unauthenticated inference endpoint. Keep the default loopback binding
or put access control in front of it; these measurements are not an Internet-facing service qualification.

## What is optimized

The checkpoint has 30 language layers: **22 short-convolution layers and 8 full-attention layers**,
32 query / 8 KV heads with head dimension 64, hidden width 2048, and a tied 128000-token embedding/head.

- **Sparse System One head:** evaluate only the required option-token rows, not all 128000 logits.
- **Shared state:** one trunk plus question branches; multiple questions do not reread their whole state.
- **Packed requests:** requests arriving while the GPU is busy share the next forward. No generated-token loop.
- **Tree attention:** causal attention within branches plus attention to the common trunk, merged using
  log-sum-exp. Convolution predecessor indices prevent one branch from reading another.
- **CUDA graphs:** short single-question text requests use fixed right-padded buckets through 224 tokens.
  Captured indices remain alive with each graph. Larger, multi-question and image requests use the eager path.
- **Fused Triton operations:** residual-add/RMSNorm, SiLU×up, tree short convolution, and query/key norm+RoPE.
- **Shape-dependent projection kernels:** quantized lane matmuls for short inputs; a BF16 reconstruction of
  the same quantized weights through cuBLAS at 256 or more packed rows.

During development, fusing the elementwise work reduced a 3446-token BF16 forward from 379.4 to 254.8 ms
in an in-process A/B run. Those development numbers are not the final HTTP benchmark. BF16 roundoff is
not bit-exact across kernel implementations; correctness evidence is probability/answer agreement, not
a claim of identical floating-point execution.

### Precision is not just a label

| Mode | Language projection weights | Activations | Caveat |
|---|---|---|---|
| BF16 | Checkpoint BF16 | BF16 | Recommended correctness baseline |
| FP8 | E4M3, FP32 scale per output row / 64 inputs | BF16 | Not INT8 Q8_0; quantization error differs |
| INT4 | Affine integers, BF16 scale+bias per 32 inputs; weighted range search | BF16 | 5 bits/weight including metadata, not Q4_K_M |

Embedding, normalization, convolution taps and vision weights remain unquantized. Quantized modes retain
BF16 projection copies for wide inputs: **packed weight size is not resident memory use**. They optimize
short-input weight traffic, not maximum memory compression. Long-prompt latency is consequently similar
across the three modes. FP8 activation GEMMs were evaluated but not shipped; they added another source
of quantization error. Round-to-nearest NVFP4 and the generic INT8 path were also evaluated and rejected.

INT4's range search was selected using the included synthetic parity corpus. That makes its reported
agreement a development-set result, **not a held-out accuracy evaluation**. Validate thresholds and
rankings on your own decisions before using any quantized mode.

## Benchmark method and results

The [complete table](results/report.md) is generated by `python3 bench/report.py results`.
Latency cells show **p50 (p99) milliseconds, CV percent**. Packed cells show median states/s and CV.
Every measured sample and each round's median are retained in the JSON files.

- Same five request bodies per engine: one question (39 prompt tokens), three questions over one state,
  a 3446-token prompt, one 384×288 picture question, and bursts of 64 distinct states.
- Warm latency measurements: five warmups, then three rounds of 50 requests: **150 samples per cell**.
- Packed measurements: five warmup bursts, then three rounds of five 64-request bursts: **15 samples**.
  Each throughput sample is 64 divided by the complete burst's wall time. This is not individual-request p99.
- The stdlib client measures complete loopback HTTP request/response latency with the same request encoding
  for both engines. A chat-completion collector is unsuitable for this zero-output-token endpoint.
- Reference: the pinned model's own `system_one` implementation through Transformers, BF16. Parity includes
  48 text requests and one image request, comprising 99 questions; packed answers are compared separately.
  Agreement is agreement with the reference, not accuracy against human labels.
- BF16 is the controlled same-model/same-precision comparison. Quantized tables compare **different formats**
  from the same source model; FP8 is not Q8_0, and affine INT4 is not Q4_K_M. Neither byte-identical quantized
  weights nor equal effective bits/weight are claimed. The BF16 image projector is used on both sides.
  All 266 language tensors were compared between the BF16 GGUF and the BF16-cast safetensors checkpoint:
  every value matched. This check does not independently audit the vision GGUF conversion.
- The three-question workload reports 116 input tokens on TensorFold versus 154 on llama.cpp: TensorFold
  counts its shared trunk once; llama.cpp counts each full question prompt. Requests themselves are unchanged.
- No application-level answer cache is used. Each backend retains its own implementation behavior; the
  experiment does not claim a separately verified zero-cache-token counter for both servers.
- The host was shared: another inference service stayed resident. Its external traffic and device utilization
  were not traced. Runs were sequential, not interleaved A/B/A bookends; these are descriptive comparisons,
  not a formal statistical regression verdict. Small differences should be treated as inconclusive.

Raw result files are named by engine/mode. [results/environment.json](results/environment.json) records
runtime/source/image identities and corpus hashes. The original downloaded picture and its resized form
are hash-checked; they are not redistributed here. Model files are pinned in [checksums/](checksums/).

### Reproduce

The shipped reference and results are retained. Use fresh labels so a new run cannot overwrite an earlier run:

```bash
./scripts/prepare.sh
# Optional: recompute into results/reference-local.json with the GPU otherwise idle.
./scripts/bench.sh reference

PRECISION=bf16 ./start.sh
./scripts/bench.sh run http://127.0.0.1:8080 trial-tensorfold-bf16
./stop.sh

LLAMACPP_PORT=8082 ./scripts/llamacpp.sh start BF16
./scripts/bench.sh run http://127.0.0.1:8082 trial-llamacpp-BF16
LLAMACPP_PORT=8082 ./scripts/llamacpp.sh stop

# Repeat with PRECISION=fp8 / int4 and baseline Q8_0 / Q4_K_M.
./scripts/bench.sh report
```

`report` reads the canonical published labels, not `trial-*`. For an individual new parity dump:

```bash
python3 bench/parity.py --compare results/reference.json results/parity-trial-tensorfold-bf16.json
```

The wrapper compares new runs against the shipped reference. To compare with a locally recomputed
reference instead, pass `results/reference-local.json` as the first `--compare` argument.

Baseline flags are fixed in `scripts/llamacpp.sh`: all layers on GPU, flash attention enabled,
64 slots, total context 262144, batch and microbatch 4096. Both engine images use the same NGC base.
If a port is occupied, override the corresponding port setting; do not stop an unrelated service.

## Verification and remaining limits

- Both containers were built and served real requests; all six benchmark runs contain their raw timings.
  The report generator recomputes summaries and rejects missing or inconsistent raw evidence.
- Final HTTP smoke covered 16 concurrent requests across both decision APIs, finite full-vocabulary mass,
  oversized single/multi-question rejection, generation refusal, and valid requests after those errors.
- Focused TensorFold tests: **162 passed, 2 skipped**, including branch-local CUDA convolution checks.
  Recipe integrity tests: **4 passed** (`python3 -m unittest discover -s tests -v`).
- A broader development run over 46 test files had the same **67 failing test IDs** on the upstream base
  and the development branch. Optional MLX was unavailable. This is not an all-green full-suite or CI claim.
- Peak memory, held-out task accuracy, long-running overload behavior, other GPUs and multi-GPU operation
  remain unqualified. Probability agreement with a small synthetic reference corpus is not a substitute.

## Pins, layout and license

- Model: `LiquidAI/d1-3B@051bcc464b01b9f92942b364d9586b0ef5912432`.
- GGUFs: `LiquidAI/d1-3B-GGUF@bb1e436ea78eb96a3f1acb6da865f70c2fbeb563`.
- TensorFold: `ed78d6fc204d89d90b045bf033d6551e7714f3a1`, plus `patches/`.
- llama.cpp: `4413a192410fad1f054e9216114087a3e01527a0` (b11580), CUDA sm_121.
- NGC PyTorch 26.07 base pinned by immutable manifest digest in `scripts/config.sh`.
- Main added Python dependencies are version-pinned there. The TensorFold image tag and label hash the
  Dockerfile, source URL/commit, base, extra packages and patches. Built image IDs are recorded with results;
  no prebuilt registry image is published by this recipe.
  The tag identifies explicit build inputs, not a byte-for-byte Docker reproducibility guarantee;
  transitive CLI/download helpers and pip build tooling are not a complete hermetic lock.

`patches/` contains the generation-tag compatibility fix, the System One HTTP route, and the LFM2 family.
`bench/` contains request generation, the HTTP client, integrity-checked parity comparison and report generator.
`reference/` calls the model's pinned upstream code (`trust_remote_code=True`); inspect that code before running
it in a different environment. `tests/test_parity.py` exercises incomplete dumps and option ordering.

Recipe and authored patches: [Apache-2.0](LICENSE). TensorFold Python 0.6: Apache-2.0. llama.cpp: MIT.
The **model is separately licensed under Liquid AI's LFM Open License v1.0**, not Apache-2.0; review its
[license](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/LICENSE).
NGC's container has NVIDIA's own terms. The recipe being open-source-ready does not relicense those
components. Weights and the COCO picture are fetched from their sources, not committed. See [NOTICE](NOTICE).
