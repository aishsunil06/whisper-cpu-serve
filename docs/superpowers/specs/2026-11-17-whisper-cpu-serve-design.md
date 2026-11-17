# whisper-cpu-serve — Design

**Date:** 2026-11-17
**Status:** Approved for planning

## Purpose

Answer a question most model demos skip: **what does it cost to run a speech
model on CPU, and what does making it cheaper cost in accuracy?**

Training a model and serving one are different skills, and the second gets far
less attention. This project deliberately trains nothing. It takes a pretrained
model, serves it, makes it faster, and proves the improvement with numbers that
survive scrutiny.

A measured latency/accuracy trade-off is a conversation. "I trained a
classifier" is a sentence.

## Success Criteria

1. A speech-to-text service reachable at a public URL, returning transcripts for
   uploaded audio.
2. A reproducible benchmark table comparing at least two engine variants on word
   error rate, latency percentiles, real-time factor, and on-disk model size.
3. Every number in that table regenerable by one command on a stated machine.
4. A defensible answer to "how did you measure that?" for each number.

Explicitly **not** a success criterion: achieving a particular speedup. A finding
that INT8 quantization costs more accuracy than it buys at this model size is a
legitimate result and must be reported as such rather than tuned away.

## Constraints

- **CPU-only.** The development machine has no CUDA (`torch 2.13.0+cpu`). This is
  not a limitation to apologize for: CPU inference cost is a real production
  concern, which makes the optimization work more relevant, not less.
- **Days, not weeks.** A deliberately tight timebox; scope is cut accordingly.
- **Free-tier hosting only.** No paid infrastructure.
- **Python 3.11 in the container**, despite 3.13 locally, for `onnxruntime` and
  `optimum` wheel availability.

## Architecture

The service and the benchmark share one interface, so the optimization claim is
measured on exactly the code that gets deployed.

```
audio.py                  bytes/file -> 16kHz mono float32; chunk long clips
engines/
  base.py                 Transcriber protocol: transcribe(audio) -> Transcript
  torch_whisper.py        baseline: HF transformers, fp32
  onnx_whisper.py         ONNX Runtime, fp32
  onnx_whisper_int8.py    ONNX Runtime, INT8 dynamic quantization
wer.py                    normalized word error rate (jiwer)
bench/
  harness.py              pinned dataset slice x engines -> metrics
  report.py               metrics -> markdown table + JSON
api.py                    POST /transcribe, GET /health
Dockerfile                -> Hugging Face Space (Docker SDK)
```

### The load-bearing decision

`Transcriber` is a protocol with a single method. Three engines implement it.
This means the benchmark varies **only the model** — identical preprocessing,
identical decode path, identical audio arrays. Without this, a latency
difference could be an accident of plumbing rather than a property of the model,
and the central claim of the project would be unfalsifiable.

A contract test runs the same assertions against every implementation,
including a `FakeTranscriber`, to keep the engines genuinely interchangeable.

### Model and data

- **Model:** `whisper-small` as baseline.
- **Data:** a 200-utterance slice of LibriSpeech test-clean, selected by fixed
  index and committed as a manifest, so every run is reproducible.

## Measurement Methodology

| | Torch fp32 | ONNX fp32 | ONNX INT8 |
|---|---|---|---|
| WER (%) | baseline | | |
| p50 / p95 latency (ms) | | | |
| Real-time factor | | | |
| Model size (MB) | | | |

Four controls make these numbers defensible:

1. **Warmup runs discarded.** First inference includes lazy initialization and
   would flatter whichever engine ran second.
2. **Thread count fixed and recorded.** CPU inference latency is meaningless
   without it.
3. **Identical preprocessed audio** passed to all engines; audio is decoded once,
   not per engine.
4. **WER normalization stated explicitly** (lowercase, strip punctuation).
   Whisper emits punctuation and LibriSpeech references do not, so unnormalized
   WER is garbage. This is the single most likely silent error in the project.

Benchmark results are recorded alongside the machine's CPU model, core count,
and thread setting. Numbers without that context are not quotable.

## Deployment

- **Container:** `python:3.11-slim` plus `ffmpeg` for audio decoding.
- **The image ships only the INT8 model.** The fp32 baselines exist locally for
  benchmarking and are never built into the container. This is the thesis of the
  project expressed as an artifact: the deployed thing is the optimized thing.
  It is also the difference between a ~300MB and a ~1.2GB image on a free tier.
- **Host:** Hugging Face Space, Docker SDK, free CPU tier, port 7860.
- **Model loads once** in a FastAPI lifespan handler. Per-request loading on a
  sleeping free-tier Space would make the first call look catastrophic.
- **Endpoints:**
  - `POST /transcribe` — multipart audio; returns transcript, audio duration,
    inference latency, engine name.
  - `GET /health` — model, engine, thread count, version, so the deployed
    service describes what is actually running.
- **Limits:** audio capped at 120s (`413`); unsupported format (`415`); model
  load failure kills startup rather than failing every request mysteriously.

### Hosted latency is not benchmark latency

Free-tier Space CPU is slow and shared. The hosted demo's latency will not match
the benchmark table, and the two claims must stay separate: quoted numbers come
from the controlled local benchmark with hardware stated; the Space exists to
prove the service is deployed and reachable. Conflating them is an
inconsistency a careful reader would catch.

## Testing

Split by speed so the fast suite stays usable during development. TDD applies:
tests first, including for the benchmark harness.

**Fast (no model loaded):**

- `audio` — 8kHz to 16kHz resample yields expected sample count; stereo collapses
  to mono; a 125s clip chunks into expected windows with correct overlap.
- `wer` — hand-computed cases. Reference "the cat sat" against hypothesis
  "The cat, sat." is exactly 0.0 after normalization; a one-substitution case is
  exactly 1/3.
- `Transcriber` contract — shared assertions parametrized over implementations.
- `api` — stub transcriber injected as a FastAPI dependency: success schema,
  415, 413, health shape. Milliseconds, no model.

**Slow (`@pytest.mark.slow`):** the real engine transcribes one short clip and
asserts WER below a loose threshold (0.25). A smoke test that the model works,
deliberately not an accuracy assertion.

**The benchmark harness is itself tested** against a fake engine with scripted
latencies, asserting that warmup runs are discarded and percentiles are computed
correctly. The harness's output is what gets quoted in interviews; an untested
measurement tool fails silently, which is worse than an untested endpoint.

## Risks

| Risk | Mitigation |
|---|---|
| **ONNX export of Whisper is fiddly.** It is an encoder-decoder with KV caching, not a single forward pass. | Use HuggingFace `optimum` (`ORTModelForSpeechSeq2Seq`), which handles the export. **Timebox to a few hours.** If it resists, fall back to comparing `faster-whisper` (CTranslate2 INT8) against the Torch baseline: same story, same metrics, different optimization mechanism. The project survives either path. |
| WER regresses badly under INT8 | Report it. This is a finding, not a failure (see Success Criteria). |
| Free-tier Space resource limits | Ship INT8 only; cap audio length; load the model once at startup. |
| Days-scale timebox slips | The out-of-scope list below is firm. |

## Out of Scope

These are what turn days into weeks:

- Streaming / real-time transcription
- Speaker diarization
- Authentication
- Any UI beyond the Space's default page
- Fine-tuning or training of any kind
- GPU inference
- Multi-language evaluation

## Reporting (post-completion)

Any claim made about this project must come from **actual measured results**,
not predicted ones. Every number in the README and RESULTS.md is produced by
`scripts/run_bench.py` on the hardware named there.
