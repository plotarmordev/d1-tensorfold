# Historical pre-audit results

These measurements describe the original packed/quantized implementation, not the current audited native-BF16
engine. FP8/INT4 are no longer supported by the current recipe. See the [current evidence](audited-bf16/summary.json).

One GB10, warm loopback HTTP. Latency: p50 (p99) milliseconds, CV percent. Packed: median states/s, CV percent.

150 latency samples per workload; 15 packed bursts per mode. Quantized formats differ. These are descriptive sequential runs on a shared host, not a statistically isolated A/B verdict. Historical recipe commit: `864a9fe`.

### BF16 checkpoints: llamacpp-BF16 vs tensorfold-bf16

| workload | llamacpp-BF16 | tensorfold-bf16 | speedup |
|---|---|---|---|
| one: one question, 39 tokens | 27.63 (37.06), 8.74% | 24.58 (25.92), 1.55% | 1.12x |
| three: three questions on one state | 119.11 (150.10), 3.78% | 28.62 (30.42), 2.25% | 4.16x |
| long: one question, 3446 prompt tokens | 447.72 (607.69), 7.78% | 270.81 (281.29), 1.46% | 1.65x |
| image: one question about a 384x288 picture | 100.28 (131.66), 4.91% | 43.35 (55.21), 5.22% | 2.31x |
| packed: 64 states at once, states/s | 36.5, CV 7.34% | 191.0, CV 1.68% | 5.24x |

| parity vs the model's own code | llamacpp-BF16 | tensorfold-bf16 |
|---|---|---|
| top answer agrees (99 questions) | 0.9899 | 0.9899 |
| median abs probability difference | 0.0045 | 0.0039 |
| p95 abs probability difference | 0.0200 | 0.0186 |
| max abs probability difference | 0.0228 | 0.0251 |
| packed set: top answer agrees (64) | 1.0000 | 1.0000 |

### 8-bit weights: llamacpp-Q8_0 vs tensorfold-fp8

| workload | llamacpp-Q8_0 | tensorfold-fp8 | speedup |
|---|---|---|---|
| one: one question, 39 tokens | 19.36 (20.17), 1.70% | 16.08 (16.89), 2.22% | 1.20x |
| three: three questions on one state | 74.17 (76.89), 1.54% | 21.98 (24.06), 2.99% | 3.37x |
| long: one question, 3446 prompt tokens | 437.12 (464.16), 1.27% | 274.71 (288.65), 1.44% | 1.59x |
| image: one question about a 384x288 picture | 76.56 (79.83), 1.79% | 41.14 (49.60), 4.18% | 1.86x |
| packed: 64 states at once, states/s | 50.7, CV 0.66% | 196.4, CV 3.58% | 3.88x |

| parity vs the model's own code | llamacpp-Q8_0 | tensorfold-fp8 |
|---|---|---|
| top answer agrees (99 questions) | 0.9899 | 0.9899 |
| median abs probability difference | 0.0055 | 0.0058 |
| p95 abs probability difference | 0.0227 | 0.0284 |
| max abs probability difference | 0.0382 | 0.0419 |
| packed set: top answer agrees (64) | 1.0000 | 0.9844 |

### 4-bit weights: llamacpp-Q4_K_M vs tensorfold-int4

| workload | llamacpp-Q4_K_M | tensorfold-int4 | speedup |
|---|---|---|---|
| one: one question, 39 tokens | 13.79 (14.85), 2.88% | 14.03 (14.95), 2.30% | 0.98x |
| three: three questions on one state | 53.58 (55.17), 1.20% | 25.14 (27.10), 2.50% | 2.13x |
| long: one question, 3446 prompt tokens | 452.13 (476.96), 1.25% | 270.90 (283.40), 1.69% | 1.67x |
| image: one question about a 384x288 picture | 61.49 (66.35), 2.17% | 48.09 (53.80), 2.81% | 1.28x |
| packed: 64 states at once, states/s | 59.2, CV 0.49% | 200.9, CV 3.43% | 3.39x |

| parity vs the model's own code | llamacpp-Q4_K_M | tensorfold-int4 |
|---|---|---|
| top answer agrees (99 questions) | 1.0000 | 0.9697 |
| median abs probability difference | 0.0181 | 0.0161 |
| p95 abs probability difference | 0.0828 | 0.0712 |
| max abs probability difference | 0.1098 | 0.0995 |
| packed set: top answer agrees (64) | 1.0000 | 0.9688 |
