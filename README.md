# d1-3B on TensorFold — DGX Spark recipe

Serve [Liquid AI's d1-3B](https://huggingface.co/LiquidAI/d1-3B) on one NVIDIA DGX Spark using
[TensorFold's Python 0.6 engine](https://github.com/ashhart/TensorFold/tree/ed78d6fc204d89d90b045bf033d6551e7714f3a1) (the pinned commit this recipe builds), not the separate native 1.x engine.
This private repository carries the audited LFM2 CUDA implementation, digest-pinned container build,
launchers, checkpoint checksums and retained performance evidence.

**Native checkpoint BF16 only.** The current engine replaces the earlier packed/quantized implementation.
It includes exact-shape CUDA graphs and the persistent-HTTP `TCP_NODELAY` fix. FP8 and INT4 are refused;
their [historical results](results/report.md) do not describe this engine.

The retained warm comparison is **INCONCLUSIVE overall**, not an unqualified performance certification.
Three-question, long-state and image gates passed; one-question reference spread and packed-throughput
uncertainty remain open. See [audited performance evidence](results/audited-bf16/summary.json).

## Quick start

Requirements: Linux, Docker with NVIDIA Container Toolkit, a CUDA-13-compatible driver, Bash, curl and
GNU coreutils. Qualified hardware coverage is one GB10 / DGX Spark, arm64, sm_121. Other GPUs and
multi-GPU operation are not qualified. Allow roughly 35 GB for the image and checkpoint.

```bash
git clone https://github.com/plotarmordev/d1-tensorfold.git
cd d1-tensorfold
./scripts/prepare.sh
./start.sh
```

Private repository access is required. `prepare.sh` builds the image and verifies the pinned checkpoint.
The build applies three source patches and verifies all eleven changed runtime files against
[checksums/tensorfold-audited.sha256](checksums/tensorfold-audited.sha256). It does not depend on a development
source mount. First startup compiles CUDA/Triton kernels into `data/cache`; later launches reuse them.
Compilation, loading and graph promotion are excluded from the warm measurements below.

```bash
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice this month, please refund one of them.",
       "questions":{"refund":{"type":"noul","instructions":"Is the customer asking for a refund?"}}}'
```

A decision is a forward pass, not generated text: responses have **zero output tokens**.
`questions` is an object keyed by question ID, not a list. Chat/completion generation is refused.

| Type | Criteria | Result |
|---|---|---|
| `noul` | Optional true/false descriptions | Probability of yes |
| `choice` | Object mapping option names to descriptions | Chosen name, confidence, probabilities |
| `score` | List of 2–10 ordered descriptions | Expected numeric level, confidence, probabilities, legend |

Confidence follows the model's native best-option probability, not a substitute entropy formula.
For images, supply `images: ["data:image/jpeg;base64,..."]` with textual `state` or `state: null`.
The shared image parser defaults to four images; remote URLs and video are unsupported. The engine uses
the checkpoint's SigLIP2 tower and projector. The retained performance image is one 384×288 picture,
not a qualification of arbitrary images or multi-image quality.

`/v1/decisions` is a separate schema, not a System One alias. Its probability-mass contract uses the full
vocabulary head; System One uses the sparse option head. `/health` and `/v1/models` are available.

## Settings and memory

Copy `scripts/local.sh.example` to git-ignored `scripts/local.sh` for local settings.
Caller/environment values override that file, which overrides [defaults](scripts/config.sh).
Models, caches, downloaded pictures and local settings stay outside git.

| Setting | Default | Meaning |
|---|---|---|
| `PRECISION` | `bf16` | Native BF16 only; other values fail before replacing a server |
| `CONTEXT` | `32768` | Serving context, 1–32768; retained performance measurements used 4096 |
| `HOST`, `PORT` | `127.0.0.1`, `8080` | Endpoint binding |
| `DATA_DIR` | `./data` | Model downloads and caches |
| `TOKEN_BUDGET` | `65536` | Trunk-plus-branch token limit; effective limit is capped by admitted context |
| `CUDA_GRAPHS` | `1` | Set `0` for eager execution |
| `MIN_FREE_GIB` | `56` | Required host MemAvailable before launch, not a runtime cap |

```bash
CONTEXT=4096 ./start.sh       # context used in the retained performance comparison
./stop.sh                    # stops this recipe's TensorFold and historical baseline containers
```

GB10 has unified CPU/GPU memory. The audited candidate's measured system-memory increase was **8.94 GiB
loaded** and **16.69 GiB after text, image and 64-client workloads**, at 4096 context. These are approximate
shared-host snapshots, not peaks; GPU allocation and CPU RSS are components, not additional memory to sum.
Full-32K-context peaks and minimum RAM remain unqualified. The graph cache's 2 GiB workspace estimate is
not a process-memory ceiling. The conservative 56 GiB startup floor leaves room for temporary allocations
and other processes; do not lower it without measuring your own workload.

A prior CUDA context creation failed before weight loading despite substantial MemAvailable. Admission
cannot guarantee allocation succeeds. Do not reset GPUs or stop unrelated services as automatic recovery.
Reserve the GPU operationally and avoid overlapping model launches. Containers do not auto-restart.

`HOST=0.0.0.0` exposes an unauthenticated endpoint. Keep loopback binding or provide access control.

## What is optimized

- Native BF16 checkpoint tensors: no on-load quantization, casts or alternate projection copies.
- Sparse System One output head: only required option-token rows, not all 128000 vocabulary entries.
- Shared state trunk and question branches within one request. Tree attention and branch-local convolution
  preserve question isolation. A lock serializes GPU requests; there is no cross-request batcher or answer cache.
- Fused residual/RMS normalization, SiLU multiplication, short convolution and query/key normalization with RoPE.
- Exact-shape CUDA graphs keyed by trunk length and all branch lengths, without padding or alternate math.
  Replay replaces every input embedding, including vision inputs, and copies outputs before reuse.
- Bounded graph roster: at most eight resident graphs and a 2 GiB estimated workspace budget, with eviction.
  First sightings are eager; promotion performs two warmup forwards, capture and replay. Rare shapes stay eager.
- HTTP `TCP_NODELAY`: avoids delayed-ACK stalls caused by separate header/body writes on reused connections.
  Graphs alone did not resolve the observed roughly 41 ms transport penalty.

The architecture has 30 language layers: 22 short-convolution and eight attention layers, hidden width 2048,
32 query / eight KV heads, head dimension 64 and a tied 128000-token embedding/head.

## Retained performance and fidelity evidence

The final frozen A/B/A2 comparison used llama.cpp BF16 / audited TensorFold / identical llama.cpp BF16,
sequential fresh servers, a shared GB10 host and 4096 context. Another inference service remained resident;
its external traffic was not controlled. Means are descriptive; verdicts use the original observed-envelope gates.

| Workload | Reference A | TensorFold B | Reference A2 | Unit | Gate |
|---|---:|---:|---:|---|---|
| One question | 27.90 | 24.09 | 28.19 | ms | INCONCLUSIVE |
| Three questions | 118.61 | 27.36 | 119.36 | ms | PASS |
| Long state | 443.93 | 277.26 | 445.06 | ms | PASS |
| Image | 101.04 | 47.02 | 100.22 | ms | PASS |
| 64-client burst | 37.15 | 40.70 | 37.26 | requests/s | INCONCLUSIVE |

One-question reference spread was 34.15%, above the predeclared 30% guard. The packed adverse envelope
reached 6.40%, above 5% tolerance. Better means do not turn either gate into PASS.
Each arm had 2320 measured requests / 2520 decisions, plus 340 warmup requests; all 6960 measured requests
completed without request errors. Three-question usage differs: TensorFold counts the shared trunk once
(116 tokens), while the reference repeats it per question (154). Raw counters and the collector warning remain.

Cold/new-shape costs are separate: first one-question request 3.122 s, its promoting call 94.19 ms,
and long-state promotion 886.02 ms. No cold-readiness claim follows from warm timings.

The candidate's 2320 measured responses matched its earlier audited BF16 answers and usage exactly.
A separate real-checkpoint check replayed 68/68 requests exactly equal to eager answers, including images
and multiple questions; a 32-geometry pressure run exercised 30 evictions. A fixed independent CPU-FP32
reference agreed on 70/70 selected outcomes (maximum probability difference 0.012477). This is numerical
fidelity, not held-out accuracy. Earlier focused validation passed 78 host/CUDA/checkpoint tests; shared
host validation passed 167 with two skipped. These overlap and are not a full-suite-green claim.

The collector was a private unreleased TheGrill extension, not released v0.6 functionality. Its source
identity was unrecorded; its binary, plan and runtime source hashes are retained in
[provenance.json](results/audited-bf16/provenance.json). Raw payloads and original audit artifacts stay outside git;
neutralized timing derivatives and their source hashes are retained here. Failed eager and graph-only runs,
and an aborted pre-weight CUDA-context launch, remain explicitly represented in the provenance record.
No new llama.cpp comparison is required to run this recipe or its direct intelligence pilot.

## Run checks on your own host

The baked-image [integration smoke](results/audited-bf16/integration.json) completed 20 qualifier requests,
five graph captures and 15 replays with no failures. The actual launcher also passed persistent-connection
and 16-client checks, zero output tokens, oversized/generation refusal, and fail-before-replacement checks.
Four recipe integrity tests passed. No hosted CI workflows are configured; this is local verification.

```bash
python3 -m unittest discover -s tests -v
./scripts/prepare.sh
# Run the live qualifier in the built image with the checkpoint and a writable output directory mounted:
# python3 bench/qualify_recipe.py --model /model --context 4096 --output /output/qualification.json --label bf16
./scripts/bench.sh run http://127.0.0.1:8080 trial-native-bf16
```

Use fresh result labels: existing evidence is never overwritten. The benchmark wrapper compares against the
shipped historical Transformers reference; agreement is not task accuracy. The live qualifier checks repeated
and concurrent answers, zero generated tokens, images and actual graph-cache activity.

[Historical results](results/report.md) and their raw JSON remain unchanged, except their report is explicitly
labeled pre-audit. `python3 bench/report.py results` regenerates that historical table, not the table above.
The historical llama.cpp helper remains available for reproduction (`LLAMACPP_PORT`, `CUDA_ARCH`); it is not
part of the current direct D1-versus-Jev pilot. Old FP8/INT4 modes require the historical recipe commit,
not `PRECISION=fp8` or `PRECISION=int4` on this branch.

## Pins and licensing

- Model: `LiquidAI/d1-3B@051bcc464b01b9f92942b364d9586b0ef5912432`.
- TensorFold: `ed78d6fc204d89d90b045bf033d6551e7714f3a1` plus the three audited patches.
- Patched-package SHA256 manifest: `69e14f6cbb17bc99bd377d2178cab6a74e2b1a062b439b7a567ffb19e87f1bed`.
- NGC PyTorch 26.07 base: immutable manifest digest in `scripts/config.sh`.
- Historical GGUF pin: `bb1e436ea78eb96a3f1acb6da865f70c2fbeb563`; llama.cpp: `4413a192410fad1f054e9216114087a3e01527a0`.

The image hash covers the Dockerfile, source URL/commit, base image, explicit extra dependencies, patches
and runtime source manifest. It identifies build inputs, not byte-for-byte hermetic Docker reproducibility;
transitive build/download tooling is not completely locked. No registry image is published.

Recipe and authored patches: [Apache-2.0](LICENSE). TensorFold Python 0.6: Apache-2.0. Historical llama.cpp: MIT.
The model is separately licensed under [Liquid AI's LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/LICENSE).
NGC's container has NVIDIA's own terms. These licenses are not replaced by the recipe license.
Weights and downloaded benchmark images are not redistributed. See [NOTICE](NOTICE).
