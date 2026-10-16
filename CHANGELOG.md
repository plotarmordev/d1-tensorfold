# Changelog

## Calibration tooling

- Added `bench/calibration.py`: fetches the Fast Decisions dev split (`fastino/fast-decisions@35afb79a`, Apache 2.0)
  into the git-ignored cache and checks it against `checksums/fast-decisions.sha256`, sends every single-label head
  to `/v1/systemone` as a `choice` over its labels, and scores accuracy and top-label ECE (10 equal-width bins,
  weighted by count) with seeded 1000-sample row-bootstrap 95% intervals.
- Each run records what served it. Before the first question, `docker inspect` of the recipe container gives the
  image id, the command and context, and the precision, token budget, CUDA-graph and TF32 settings. The mounted
  checkpoint is checked against `checksums/d1-3B.sha256`, and its `hf download` revision is recorded.
  - This only applies to a loopback URL that the container publishes, through a local Docker daemon.
  - A run stops if any of this is missing, or if the container changes during the run.
  - Any other server needs `--provenance FILE`, which records operator-supplied evidence.
- Added `bench/calibration_chart.py`, a standard-library SVG reliability diagram and accuracy-minus-confidence chart,
  and tests for the binning, ECE and bootstrap and for a full run against a stub server. No calibration result yet.

## Audited native-BF16 cutover

- Replaced the earlier packed/quantized family with the measured native-checkpoint BF16 implementation:
  serialized requests, within-request shared trunks, sparse System One heads, fused kernels and exact-shape graphs.
- Included HTTP `TCP_NODELAY`, eliminating the measured persistent-connection delayed-ACK stall.
- Removed active FP8/INT4 and padded-graph paths. Historical results remain explicitly labeled pre-audit.
- Added byte verification of all eleven patched runtime files during image build and included the manifest in
  the image-input hash. No development source bind mount is required.
- Added explicit serving context and fail-before-replacement validation, native TF32 override settings,
  and the real-checkpoint repeated/concurrent/image qualifier.
- Retained all final timing waves as neutralized, hash-pinned derivatives, memory snapshots and provenance.
  Overall performance qualification remains INCONCLUSIVE; warm means, cold costs and open gates remain distinct.

## 2026-10-13

- Added single-GB10 deployment of LiquidAI/d1-3B through TensorFold's Python 0.6 engine, as three source patches.
- Implemented the System One HTTP API, BF16 vision, typed question scoring and zero-token responses.
- Added shared-prefix tree execution, queued request packing, a sparse option-token head, short-request CUDA
  graphs, fused Triton operations, and BF16 / block-FP8 / affine INT4 projection modes.
- Added per-sequence and packed-forward bounds; serialized generic decision scoring with the worker. Generic
  decisions compute full-vocabulary mass rather than treating the label-only normalizer as vocabulary mass.
- Added digest-pinned base image, source and model pins, model checksums, content-addressed image builds,
  user-owned kernel caches, launch memory preflight and explicit start/stop scripts.
- Added a pinned llama.cpp baseline, deterministic text/image workloads, all raw warm samples, probability
  parity against the model's own implementation, and report generation.
- Removed experimental NVFP4 and generic INT8 modes after speed/quality evaluation. INT4 is lossy; its selected
  quantizer was tuned on the included development corpus. Quantized comparisons are not format-identical.
- Model weights, downloaded benchmark pictures and local machine settings are not redistributed.
