# Changelog

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
