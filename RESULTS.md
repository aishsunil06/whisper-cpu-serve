# Benchmark Results

Model: `openai/whisper-base` | Generated: 2026-11-17T18:39:13.944069+00:00
Warmup runs discarded per engine: 2

| Engine | WER | p50 latency | p95 latency | RTF | Model size | Speedup | Size reduction |
|---|---|---|---|---|---|---|---|
| `torch_fp32` | 9.5% | 1491 ms | 2714 ms | 0.26 | 290 MB | 1.00x | 1.00x |
| `onnx_fp32` | 9.5% | 883 ms | 4180 ms | 0.22 | 699 MB | 1.69x | 0.42x |
| `onnx_int8` | 12.0% | 764 ms | 1520 ms | 0.14 | 178 MB | 1.95x | 1.63x |

Evaluated on **73 utterances** (LibriSpeech validation-clean).

Host: Intel64 Family 6 Model 186 Stepping 2, GenuineIntel | 16 logical cores | 4 inference threads | Windows-11-10.0.26200-SP0

Real-time factor (RTF) is inference seconds per second of audio; below 1.0 is faster than real time. Latency excludes audio decoding, which is identical across engines, and excludes warmup runs.
