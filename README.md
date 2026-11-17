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

<!--RESULTS-->

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
pip install -r requirements.txt
python scripts/export_onnx.py          # exports fp32 + INT8 graphs to models/
ENGINE=onnx_int8 PYTHONPATH=src uvicorn whisper_cpu_serve.api:app --port 7860
```

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
  "engine": "onnx_int8"
}
```

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

```bash
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
