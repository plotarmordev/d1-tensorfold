# d1-3B on TensorFold

**Serve Liquid AI's d1-3B decision model on one DGX Spark, with the evidence to check it.**

[d1-3B](https://huggingface.co/LiquidAI/d1-3B) answers questions about a state (text, a picture, or both)
with probabilities, not generated text. This recipe serves it through
[TensorFold's Python 0.6 engine](https://github.com/ashhart/TensorFold/tree/ed78d6fc204d89d90b045bf033d6551e7714f3a1)
at a pinned commit plus three source patches, implementing llama-server's System One API. A decision is one forward
pass: every response has **zero output tokens**.

| I want to... | Use | What you get |
|---|---|---|
| Serve d1-3B | `./start.sh` | `POST /v1/systemone` and `/v1/decisions` on `127.0.0.1:8080` |
| Measure a running server | `scripts/bench.sh run` | Latency, throughput and parity against the model's own code, saved under `results/` |
| Compare with llama.cpp | `scripts/llamacpp.sh start` | llama-server at a pinned commit, serving Liquid AI's GGUF files on `127.0.0.1:8081` |

Everything runs on your machine. The weights are downloaded from Hugging Face at a pinned revision and checked
against [checksums](checksums/d1-3B.sha256); they are never part of this repository.
**Native BF16 only, qualified on one GB10.**

## Stock llama.cpp vs this build

Same GB10, warm loopback HTTP, 4096 context. Stock is llama.cpp serving the model in BF16; ours is the audited
TensorFold implementation integrated into this recipe, also BF16. Three fresh servers ran one after another:
stock (A), ours (B), then stock again (A2) as a control. Values are means; the ratio compares ours with each
stock run (stock latency over ours; for the burst, ours over stock).

| Workload | Stock A | Stock A2 | Ours | Ours vs stock | Verdict | Source |
|---|---:|---:|---:|---:|---|---|
| One question, 39 input tokens | 27.90 ms | 28.19 ms | 24.09 ms | 1.16–1.17× faster | INCONCLUSIVE | [summary](results/audited-bf16/summary.json) |
| Three questions on one state | 118.61 ms | 119.36 ms | 27.36 ms | 4.34–4.36× faster | PASS | [summary](results/audited-bf16/summary.json) |
| One question, 3446 input tokens | 443.93 ms | 445.06 ms | 277.26 ms | 1.60–1.61× faster | PASS | [summary](results/audited-bf16/summary.json) |
| One question about a 384×288 picture | 101.04 ms | 100.22 ms | 47.02 ms | 2.13–2.15× faster | PASS | [summary](results/audited-bf16/summary.json) |
| 64-client burst | 37.15 req/s | 37.26 req/s | 40.70 req/s | 1.09–1.10× more | INCONCLUSIVE | [summary](results/audited-bf16/summary.json) |
| Input tokens, three-question request | 154 | 154 | 116 | 24.68% fewer reported input tokens | n/a | [timings](results/audited-bf16/timings.json) |

Not compared, because the repository has no matching record:

- **Decode tokens/s.** Both servers answer with zero output tokens ([timings](results/audited-bf16/timings.json)),
  so there is no decode phase. Prefill and time to first token are not recorded separately; the latency above
  covers the whole decision request.
- **Memory.** Only ours was measured: system memory rose by 8.94 GiB once loaded and by 16.69 GiB after the
  text, image and 64-client workloads ([memory](results/audited-bf16/memory-summary.json)). These are snapshots
  on shared unified memory, not peaks. No stock llama.cpp memory figure was recorded.
- **Stock Q8_0 and Q4_K_M.** These were measured only against the earlier packed/quantized engine, which this
  build replaced, in a separate session. Those [historical results](results/report.md) do not describe this build.

The overall verdict is **INCONCLUSIVE**, not a performance certification: the two one-question control runs
spread too far apart to call, and the burst's adverse bound reached 6.40%, past its tolerance. A better mean
does not turn either gate into a pass. The summary also flags changed reported input-token exposure for the
three-question state: 116 for TensorFold versus 154 for llama.cpp. Each run answered 2320 measured requests
(2520 decisions) after 340 warmup requests, all complete. These timings precede packaging; the baked image
passed integration checks, not a new performance comparison.

Several questions on one state are where the recipe gains most: when the questions fit in one forward, the state
is read once and each question branches off it, which is why TensorFold reports 116 input tokens for the
three-question request where llama.cpp reports 154.

The packaged image's integration checks recorded repeated/concurrent response agreement and zero output tokens.
The retained timing derivatives omit answer payloads; these checks do not establish held-out accuracy.

Evidence: [summary](results/audited-bf16/summary.json), [timings](results/audited-bf16/timings.json),
[memory](results/audited-bf16/memory-summary.json), [provenance](results/audited-bf16/provenance.json).

## Requirements

- One NVIDIA DGX Spark (GB10, arm64, compute capability 12.1). Other GPUs and multi-GPU setups are not qualified.
- Linux, Docker with the NVIDIA Container Toolkit, and a CUDA-13-compatible driver.
- Bash, curl, GNU coreutils and Git; Python 3 for the optional host-side unit tests.
- 56 GiB of available memory at launch. The launchers refuse to load below that (`MIN_FREE_GIB`).
- Access to Hugging Face, NVIDIA NGC and GitHub for the first build and download. Set `HF_TOKEN` if your
  Hugging Face account needs it.

## Quickstart

```bash
git clone https://github.com/plotarmordev/d1-tensorfold.git
cd d1-tensorfold
./start.sh
```

`start.sh` runs `scripts/prepare.sh` first. That builds the image, applies the three patches and verifies all
eleven changed runtime files against [checksums/tensorfold-audited.sha256](checksums/tensorfold-audited.sha256),
then downloads d1-3B and verifies every file. The first start compiles CUDA and Triton kernels into
`data/cache`; later starts reuse them.

Ask a question:

```bash
curl -s http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"state":"I was charged twice this month, please refund one of them.",
       "questions":{"refund":{"type":"noul","instructions":"Is the customer asking for a refund?"}}}'
```

`questions` is an object keyed by question ID, not a list. Each question has one of three types:

| Type | Criteria | Answer |
|---|---|---|
| `noul` | Optional true/false descriptions | Probability of yes |
| `choice` | Object mapping option names to descriptions | Chosen name, confidence, probabilities |
| `score` | List of 2 to 10 ordered descriptions | Expected level, confidence, probabilities, legend |

For pictures, send `images: ["data:image/jpeg;base64,..."]` with a text `state` or `state: null`. Up to four
pictures by default; remote URLs and video are not supported. `/v1/decisions` is a separate schema, not an alias.
`/health` and `/v1/models` are available; chat and completions are refused. `./stop.sh` stops the server.

### Settings

Copy `scripts/local.sh.example` to `scripts/local.sh` (git-ignored). Environment variables override that file,
which overrides the [defaults](scripts/config.sh). Precision, ports, context, token budget, graph mode and the
memory floor are checked before a running server is replaced; not every setting is prevalidated.

| Setting | Default | Meaning |
|---|---|---|
| `CONTEXT` | `32768` | Serving context, 1 to 32768. The measurements used 4096 |
| `HOST`, `PORT` | `127.0.0.1`, `8080` | Endpoint binding |
| `DATA_DIR` | `./data` | Model downloads and kernel caches |
| `TOKEN_BUDGET` | `65536` | Tokens per forward, 4096 to 65536; the effective budget is capped by `CONTEXT`, and larger question sets are split across forwards |
| `CUDA_GRAPHS` | `1` | `0` runs eagerly |
| `MIN_FREE_GIB` | `56` | Available memory required before launch, not a runtime cap |
| `PRECISION` | `bf16` | Native BF16 only; anything else is refused |

## Reproduce

The recipe's own checks, without a GPU:

```bash
python3 -m unittest discover -s tests -v
```

Measure this recipe, then the llama.cpp baseline. Run the servers one at a time, as the measurements did:

```bash
CONTEXT=4096 ./start.sh
./scripts/bench.sh run http://127.0.0.1:8080 my-tensorfold-bf16
./stop.sh

./scripts/llamacpp.sh start BF16          # or Q8_0, Q4_K_M
./scripts/bench.sh run http://127.0.0.1:8081 my-llamacpp-BF16
./scripts/llamacpp.sh stop
```

Each run writes `results/LABEL.json` and `results/parity-LABEL.json` and refuses to overwrite existing results,
so use a fresh label. Parity compares against the shipped Transformers reference
([results/reference.json](results/reference.json)); agreement is not task accuracy.
`./scripts/bench.sh reference` regenerates the reference with the model's own code on the GPU and writes
`results/reference-local.json` by default; `./scripts/bench.sh report` prints the historical tables to stdout.
Neither overwrites the shipped [reference](results/reference.json) or [report](results/report.md).

The live qualifier checks repeated/concurrent answer agreement and zero output tokens, exercises pictures, and
records CUDA graph statistics. It loads its own engine and server, so stop the serving containers first. It is an
in-container command: run it in the built image with GPU access, the repository mounted at `/recipe` as the working
directory, the checkpoint at `/model`, a writable `/output` and a writable cache at `/cache`:

```bash
python3 bench/qualify_recipe.py --model /model --context 4096 --output /output/qualification.json --label bf16
```

The image's [integration run](results/audited-bf16/integration.json) recorded 20 qualifier requests and no
failures; its graph statistics recorded five captures and 15 replays.

## Limitations

- **The overall performance verdict is INCONCLUSIVE.** Three of five workloads passed; the one-question and
  burst gates did not.
- **Warm numbers only.** Cold and new-shape costs are separate: the first one-question request took 3.122 s,
  the call that promoted its graph 94.19 ms, and the long-state promotion 886.02 ms.
- **Shared host.** Another inference service stayed resident during the measurements, and its traffic was not
  controlled.
- **Memory is not fully qualified.** Snapshots were taken at 4096 context; peaks at the full 32768 context and the
  minimum RAM are unknown. A launch can still fail to allocate: one earlier attempt failed to create its CUDA
  context before loading weights. Containers do not restart on their own.
- **BF16 only.** FP8 and INT4 are refused. Their historical numbers need the historical recipe commit `864a9fe`.
- **The A/B/A2 collector is unreleased.** The collector binary and plan hashes are recorded in
  [provenance.json](results/audited-bf16/provenance.json); the collector source revision was not recorded.
- **Not hermetic.** The image hash covers the declared build inputs, but transitive build tooling is not locked.
  No registry image is published.
- **No hosted CI.** All verification is local.
- `HOST=0.0.0.0` exposes an unauthenticated endpoint. Keep loopback binding or add access control.

## License and credits

The recipe and its patches are licensed under [Apache 2.0](LICENSE). It builds on:

- [TensorFold](https://github.com/ashhart/TensorFold) `ed78d6fc204d89d90b045bf033d6551e7714f3a1`, Apache 2.0.
- [llama.cpp](https://github.com/ggml-org/llama.cpp) `4413a192410fad1f054e9216114087a3e01527a0`, MIT, for the baseline.
- [d1-3B](https://huggingface.co/LiquidAI/d1-3B) by Liquid AI, revision `051bcc464b01b9f92942b364d9586b0ef5912432`
  (GGUF `bb1e436ea78eb96a3f1acb6da865f70c2fbeb563`), under the
  [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/LICENSE).
- NVIDIA's NGC PyTorch 26.07 container, under NVIDIA's terms; its digest is pinned in `scripts/config.sh`.
- The benchmark picture, COCO val2017 image 39769, fetched at bench time under its owner's license.

Weights and benchmark pictures are not redistributed. Their licenses are not replaced by the recipe's.
See [NOTICE](NOTICE).
