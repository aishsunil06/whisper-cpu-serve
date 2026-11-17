---
title: Whisper CPU Serve
emoji: 🎙️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# whisper-cpu-serve

Speech-to-text served on CPU, with the inference cost actually measured.

This project takes a pretrained Whisper model, serves it behind an HTTP API,
and asks a question most model demos skip: **what does it cost to run, and what
does making it cheaper cost in accuracy?**

The answer is a benchmark comparing three engines over the same audio, and a
deployed container that ships only the fastest one.

## Results

`openai/whisper-base`, 73 utterances of LibriSpeech validation-clean, 2 warmup
runs discarded per engine, 4 inference threads on an Intel Core (16 logical
cores), Windows 11.

| Engine | WER | p50 latency | p95 latency | RTF | Model size | Speedup | Size reduction |
|---|---|---|---|---|---|---|---|
| `torch_fp32` | 9.5% | 1491 ms | 2714 ms | 0.26 | 290 MB | 1.00x | 1.00x |
| `onnx_fp32` | 9.5% | 883 ms | 4180 ms | 0.22 | 699 MB | 1.69x | 0.42x |
| `onnx_int8` | 12.0% | 764 ms | 1520 ms | 0.14 | 178 MB | 1.95x | 1.63x |

Regenerate with `python scripts/run_bench.py`; raw output lands in
`bench/results/latest.json`.

### What the numbers say

**INT8 quantization bought 1.95x on median latency and 1.63x on model size, and
cost 2.5 points of WER.** 9.5% to 12.0% is a 26% relative increase in errors.
Whether that is a good trade is a product question, not a benchmark question: it
is clearly worth it for a draft transcript a human will edit, and clearly not
worth it for anything where the transcript is the final artifact.

**ONNX fp32 shows a worse tail than Torch despite a better median** — p95 of
4180 ms against Torch's 2714 ms, on a better p50. If that holds, it means a
median-only comparison would have called ONNX fp32 a straight 1.69x win and been
misleading, which is the argument for reporting percentiles rather than means.

**This specific observation is not yet verified.** The table comes from a single
run with engines benchmarked in one contiguous block each, in the order
`torch_fp32 → onnx_fp32 → onnx_int8`, on a thermally-managed laptop. Block
ordering is a plausible confound for tail latency specifically, and nothing in
the harness separates a model property from a scheduling artifact in the second
block. Treat the p50, WER and size columns as solid and this p95 observation as
provisional until a repeated or interleaved run confirms it. The p50 speedup and
the WER cost are not ordering-sensitive in the same way.

**ONNX fp32 is also 2.4x larger than the Torch model it was exported from**
(699 MB vs 290 MB). The export emits `decoder_model.onnx` and
`decoder_with_past_model.onnx` as separate graphs, each carrying a full copy of
the decoder weights. Quantization is what pulls the served artifact back under
the original (178 MB) — so on this pipeline, INT8 is not merely an optimization,
it is what makes the ONNX path worth deploying at all.

**Every engine ran faster than real time** (RTF below 1.0), so even the baseline
could serve this workload. The optimization buys headroom and concurrency, not
feasibility.

## Why this is set up the way it is

**One interface, three engines.** `Transcriber` is a protocol with a single
`transcribe` method. Torch fp32, ONNX fp32 and ONNX INT8 all implement it, and
the benchmark drives all three through identical, already-decoded audio. Without
that, a latency difference could be an artifact of the plumbing rather than a
property of the model, and the central claim here would be unfalsifiable.

**Audio decoding happens once, outside the timer.** Preprocessing is identical
across engines, so timing it would dilute the difference being measured.

**Warmup runs are discarded.** The first inference pays for lazy graph
initialization. Counting it attributes startup cost to the model and invents a
speedup that is not real — the single easiest way to accidentally fabricate a
benchmark result.

**The benchmark harness has its own tests.** It is verified against a fake engine
with scripted latencies, which is how the warmup exclusion and percentile maths
are proven rather than assumed. An untested measurement tool fails silently.

**The container ships only the INT8 graphs.** The fp32 exports exist locally for
comparison and are excluded via `.dockerignore`. The deployed artifact is the
optimized one; that is the whole point.

## Honest limitations

- **Hosted latency is not benchmark latency.** The free-tier Space runs on
  shared, throttled CPU. Numbers in the table above come from the controlled
  local benchmark on the hardware named there. Do not compare the two.
- **The evaluation set is small** — LibriSpeech validation-clean via
  `hf-internal-testing/librispeech_asr_dummy`, read-aloud audiobook speech. It is
  clean, American-accented, and not representative of noisy or accented audio.
- **`whisper-base`, not `whisper-small`.** Chosen so a full three-engine run
  finishes fast enough to iterate on. Raise `MODEL_ID` to re-run at `small`.
- **No training or fine-tuning.** This project is about serving and optimizing a
  model, not producing one.
- **Dynamic quantization only.** Static quantization needs a calibration set and
  was out of scope.
- **The container is 2.16GB, and quantization barely dents that.** Shipping INT8
  instead of fp32 graphs saves ~520MB of weights, but `torch`, `transformers` and
  `onnxruntime` dominate the image regardless. Model-size reduction is a real
  win for memory and load time, not for image size; slimming the image further
  would mean dropping `torch`, which `optimum` requires.

## Usage

### Run the API locally

```bash
# On Python 3.13 add --only-binary=:all: -- pip otherwise tries to source-build
# an older numpy pin and fails.
pip install -r requirements-dev.txt
python scripts/export_onnx.py          # exports fp32 + INT8 graphs to models/
ENGINE=onnx_int8 PYTHONPATH=src uvicorn whisper_cpu_serve.api:app --port 7860
```

`requirements.txt` is serving-only (what the container installs);
`requirements-dev.txt` adds the benchmark and test dependencies.

```bash
curl localhost:7860/health
curl -F file=@clip.wav localhost:7860/transcribe
```

`POST /transcribe` returns:

```json
{
  "text": "transcribed speech",
  "audio_seconds": 2.0,
  "latency_ms": 412.3,
  "engine": "onnx_int8",
  "n_chunks": 1
}
```

Audio longer than 30 seconds is split into non-overlapping 30-second windows,
because Whisper's feature extractor silently truncates anything longer.
`n_chunks` reports how many windows were used, so a caller can tell windowing
happened instead of inferring it from a short transcript. A word landing exactly
on a window boundary may be split.

`GET /health` reports the model, engine, and thread count actually running, so
a deployment can be verified rather than assumed.

Audio over 120 seconds returns `413`; undecodable files return `415`; empty
uploads return `400`.

### Reproduce the benchmark

```bash
python scripts/run_bench.py --engines torch_fp32,onnx_fp32,onnx_int8
```

Writes `RESULTS.md` and `bench/results/latest.json`. Every number is measured by
that command; none are estimated.

### Run the tests

```bash
pytest           # fast suite, loads no model
pytest -m slow   # also runs real models against a real clip
```

### Build the container

The exported INT8 graphs are not committed (they are build artifacts), so
generate them before building:

```bash
python scripts/export_onnx.py          # required: Dockerfile copies models/onnx-int8/
docker build -t whisper-cpu-serve .
docker run -p 7860:7860 whisper-cpu-serve
```

## Layout

```
src/whisper_cpu_serve/
  audio.py              decode -> 16kHz mono float32; chunking
  wer.py                normalized, corpus-aggregated word error rate
  config.py             model id, thread count, limits
  api.py                FastAPI service; engine injected for testability
  engines/
    base.py             Transcriber protocol + Transcript
    fake.py             no-model engine for testing the API and the harness
    torch_whisper.py    stock PyTorch fp32 baseline
    onnx_whisper.py     ONNX Runtime, fp32 and INT8
  bench/
    harness.py          timing, percentiles, warmup exclusion, RTF
    dataset.py          evaluation set, decoded through audio.py
    report.py           markdown table rendering
scripts/
  export_onnx.py        ONNX export + dynamic INT8 quantization
  run_bench.py          benchmark CLI
```
