# whisper-cpu-serve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve a Whisper speech-to-text model behind an HTTP API, measure CPU inference cost against accuracy across model variants, and deploy only the optimized variant.

**Architecture:** A `Transcriber` protocol with one method; several engine implementations behind it; a benchmark harness that drives the same preprocessed audio through each engine so only the model varies. The API loads one engine, chosen by environment variable, once at startup.

**Tech Stack:** Python 3.11 (container) / 3.13 (local), PyTorch CPU, HuggingFace `transformers` + `optimum[onnxruntime]`, `onnxruntime`, `soundfile` + `soxr` for audio, `jiwer` for WER, FastAPI + uvicorn, pytest, Docker.

**Spec:** `docs/superpowers/specs/2026-11-17-whisper-cpu-serve-design.md`

## Global Constraints

- CPU-only inference. No CUDA code paths, no `.cuda()`, no device arguments.
- No `ffmpeg`. Audio decoding is `soundfile` (libsndfile) only; this keeps the container lean and was verified absent locally.
- Target sample rate is exactly **16000 Hz, mono, float32** everywhere downstream of `audio.py`.
- Audio longer than **120 seconds** is rejected by the API with HTTP **413**.
- Unsupported/undecodable audio is rejected with HTTP **415**.
- WER normalization is **lowercase + strip punctuation**, applied identically to reference and hypothesis.
- Benchmark results must record CPU model, core count, and thread count alongside every measurement.
- The deployed container ships the INT8 model only.

### Deviations from the spec, decided during planning

1. **Model is `whisper-base`, not `whisper-small`.** Three engines over a benchmark set on CPU with `small` makes each full run tens of minutes, which kills iteration inside a days-scale timebox. `base` preserves every part of the story. The model id is a constant in one place so it can be raised later.
2. **Dataset is `hf-internal-testing/librispeech_asr_dummy` (73 utterances from LibriSpeech validation-clean), not a 200-utterance test-clean slice.** Full test-clean is a ~346MB shard; the dummy set is ~9MB and is the standard fixture for exactly this comparison. The eval set size must be stated honestly in the README and anywhere the numbers are quoted.

## Review Focus

Input classes the spec implies but no happy-path test covers. Each has its test assigned to the task owning the code.

1. **Zero-byte or empty upload** — a 0-length file must return a 4xx, not crash the decoder (Task 8).
2. **Non-audio bytes with an audio filename** — a text file named `.wav` must return 415, not a 500 traceback (Task 8).
3. **Non-16kHz input** — 8kHz and 44.1kHz audio must resample to the correct sample count, not produce chipmunked garbage (Task 1).
4. **Empty reference string in WER** — an empty reference must not divide by zero (Task 2).
5. **Audio exactly at and just over the 120s limit** — the boundary must be inclusive/exclusive deliberately, not accidentally (Task 8).

---

### Task 1: Audio loading and preprocessing

**Files:**
- Create: `src/whisper_cpu_serve/audio.py`
- Create: `tests/test_audio.py`
- Create: `pyproject.toml`, `.gitignore`

**Interfaces:**
- Produces:
  - `load_audio(data: bytes | str | Path) -> np.ndarray` — returns 16kHz mono float32. Raises `AudioDecodeError` on undecodable input, `EmptyAudioError` on zero-length.
  - `duration_seconds(audio: np.ndarray) -> float`
  - `chunk(audio: np.ndarray, window_s: float = 30.0, overlap_s: float = 1.0) -> list[np.ndarray]`
  - Exceptions `AudioDecodeError`, `EmptyAudioError`

- [ ] **Step 1: Install dependencies and scaffold**

```bash
python -m pip install "optimum[onnxruntime]" onnxruntime jiwer soundfile soxr datasets uvicorn python-multipart
```

Create `pyproject.toml` declaring package `whisper_cpu_serve` under `src/`, and a `.gitignore` covering `__pycache__/`, `*.onnx`, `models/`, `.venv/`, `bench/results/*.json`.

- [ ] **Step 2: Write the failing tests**

```python
def test_resamples_8k_to_16k_doubling_sample_count():
    # 1.0s of 8kHz sine written to an in-memory WAV -> 16000 samples
def test_resamples_44100_to_16k():
    # 1.0s of 44.1kHz -> 16000 samples (within 1 sample)
def test_stereo_collapses_to_mono():
    # (N, 2) input -> 1-D output
def test_returns_float32_in_unit_range():
def test_duration_seconds_of_16000_samples_is_1():
def test_chunk_125s_into_30s_windows_with_1s_overlap_yields_5_chunks():
def test_chunk_shorter_than_window_returns_single_chunk():
def test_empty_audio_raises_EmptyAudioError():
def test_non_audio_bytes_raise_AudioDecodeError():
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_audio.py -v`
Expected: FAIL, module not found.

- [ ] **Step 4: Implement `audio.py`**

Decode with `soundfile.read` from a `BytesIO` for bytes input. Resample with `soxr.resample`. Collapse channels by mean along axis 1. Chunk with stride `window_s - overlap_s`, emitting a final partial chunk.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_audio.py -v`
Expected: PASS (9 tests)

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: audio loading, resampling and chunking"
```

---

### Task 2: Normalized word error rate

**Files:**
- Create: `src/whisper_cpu_serve/wer.py`
- Create: `tests/test_wer.py`

**Interfaces:**
- Produces:
  - `normalize(text: str) -> str` — lowercase, strip punctuation, collapse whitespace
  - `word_error_rate(reference: str, hypothesis: str) -> float`
  - `corpus_wer(references: list[str], hypotheses: list[str]) -> float`

- [ ] **Step 1: Write the failing tests**

```python
def test_punctuation_and_case_differences_give_zero_wer():
    assert word_error_rate("the cat sat", "The cat, sat.") == 0.0
def test_single_substitution_in_three_words_is_one_third():
    assert word_error_rate("the cat sat", "the dog sat") == pytest.approx(1/3)
def test_empty_reference_with_empty_hypothesis_is_zero():
    assert word_error_rate("", "") == 0.0
def test_empty_reference_with_nonempty_hypothesis_is_one():
    assert word_error_rate("", "hello") == 1.0
def test_corpus_wer_aggregates_over_total_words_not_mean_of_rates():
    # refs ["a b c d", "e"], hyps ["a b c d", "x"] -> 1 error / 5 words = 0.2
```

The corpus test is the one that matters: averaging per-utterance rates would give 0.5 and silently overstate accuracy on sets with short utterances.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_wer.py -v`

- [ ] **Step 3: Implement `wer.py`**

`normalize` strips via `str.translate` over `string.punctuation`. `word_error_rate` delegates to `jiwer.wer` on normalized strings, special-casing empty reference (0.0 if hypothesis also empty, else 1.0, since jiwer raises). `corpus_wer` uses `jiwer.process_words` totals so errors aggregate over all words.

- [ ] **Step 4: Run tests to verify they pass**

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: normalized WER with corpus-level aggregation"
```

---

### Task 3: Transcriber protocol and contract test

**Files:**
- Create: `src/whisper_cpu_serve/engines/base.py`
- Create: `src/whisper_cpu_serve/engines/fake.py`
- Create: `tests/test_engine_contract.py`

**Interfaces:**
- Consumes: `audio.load_audio`
- Produces:
  - `@dataclass Transcript: text: str; engine: str`
  - `class Transcriber(Protocol): name: str; def transcribe(self, audio: np.ndarray) -> Transcript; def model_size_bytes(self) -> int`
  - `FakeTranscriber(text: str = "...", latency_s: float = 0.0, name: str = "fake")` — sleeps `latency_s`, returns fixed text. Used by the harness and API tests.

- [ ] **Step 1: Write the failing contract test**

```python
ENGINES = [FakeTranscriber]  # real engines appended in Tasks 4-5

@pytest.mark.parametrize("factory", ENGINES)
def test_transcribe_returns_transcript_with_text_and_engine_name(factory):
def test_transcribe_accepts_one_second_of_silence(factory):
def test_model_size_bytes_is_positive(factory):
def test_fake_transcriber_respects_scripted_latency():
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Implement `base.py` and `fake.py`**

`Transcriber` is a `typing.Protocol` (runtime-checkable). No base class inheritance; engines satisfy it structurally.

- [ ] **Step 4: Run to verify it passes**

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: Transcriber protocol and fake engine"
```

---

### Task 4: Torch baseline engine

**Files:**
- Create: `src/whisper_cpu_serve/engines/torch_whisper.py`
- Create: `src/whisper_cpu_serve/config.py`
- Modify: `tests/test_engine_contract.py` (append to `ENGINES`)
- Create: `tests/test_torch_engine.py`

**Interfaces:**
- Consumes: `Transcriber`, `Transcript`
- Produces:
  - `config.MODEL_ID = "openai/whisper-base"`, `config.THREAD_COUNT` (from `OMP_NUM_THREADS`, default 4), `config.MAX_AUDIO_SECONDS = 120.0`
  - `TorchWhisper(model_id: str = config.MODEL_ID)` with `name = "torch_fp32"`

- [ ] **Step 1: Write the failing test** (marked `@pytest.mark.slow`)

```python
@pytest.mark.slow
def test_transcribes_a_known_clip_below_loose_wer_threshold():
    # one LibriSpeech dummy clip; assert word_error_rate(ref, hyp) < 0.25
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Implement `torch_whisper.py`**

`WhisperProcessor` + `WhisperForConditionalGeneration`, `torch.set_num_threads(config.THREAD_COUNT)`, `model.eval()`, inference under `torch.inference_mode()`. `model_size_bytes` sums `p.numel() * p.element_size()` over parameters. Register `slow` marker in `pyproject.toml`.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_torch_engine.py -v -m slow`

- [ ] **Step 5: Commit**

---

### Task 5: ONNX fp32 and INT8 engines

**Files:**
- Create: `src/whisper_cpu_serve/engines/onnx_whisper.py`
- Create: `scripts/export_onnx.py`
- Modify: `tests/test_engine_contract.py`

**Interfaces:**
- Produces:
  - `OnnxWhisper(model_dir: Path, quantized: bool = False)` with `name` of `"onnx_fp32"` or `"onnx_int8"`
  - `scripts/export_onnx.py` writing to `models/onnx-fp32/` and `models/onnx-int8/`

- [ ] **Step 1: Write the export script**

`ORTModelForSpeechSeq2Seq.from_pretrained(MODEL_ID, export=True)` saved to `models/onnx-fp32/`. Then `ORTQuantizer` with `AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)` (dynamic quantization) over each of the encoder/decoder ONNX files, saved to `models/onnx-int8/`.

**Timebox: 3 hours.** If the export or quantizer fails on this `optimum`/`transformers` pairing, stop and take the documented fallback: implement `engines/faster_whisper.py` wrapping `WhisperModel(MODEL_ID, device="cpu", compute_type="int8")` as the optimized engine and compare it against the Torch baseline. The benchmark, API, deployment and README tasks are unchanged by that substitution; only the engine list changes.

- [ ] **Step 2: Run the export**

Run: `python scripts/export_onnx.py`
Expected: both model directories exist and `models/onnx-int8` is materially smaller.

- [ ] **Step 3: Implement `onnx_whisper.py` and append both engines to the contract test's `ENGINES`**

`model_size_bytes` sums `*.onnx` file sizes in the model directory.

- [ ] **Step 4: Run the contract test across all engines**

Run: `python -m pytest tests/test_engine_contract.py -v`
Expected: PASS for fake, torch, onnx_fp32, onnx_int8.

- [ ] **Step 5: Commit**

---

### Task 6: Benchmark harness

**Files:**
- Create: `src/whisper_cpu_serve/bench/harness.py`
- Create: `tests/test_harness.py`

**Interfaces:**
- Consumes: `Transcriber`, `FakeTranscriber`, `corpus_wer`
- Produces:
  - `@dataclass EngineMetrics: engine: str; wer: float; p50_ms: float; p95_ms: float; rtf: float; model_size_mb: float; n_utterances: int`
  - `benchmark(engine, samples: list[Sample], warmup: int = 2) -> EngineMetrics`
  - `@dataclass Sample: audio: np.ndarray; reference: str`
  - `host_info() -> dict` — CPU model, physical/logical cores, thread count

This task's tests are the ones that protect every number quoted from this project, so they are written against a fake engine with scripted latencies rather than a real model.

- [ ] **Step 1: Write the failing tests**

```python
def test_warmup_runs_are_excluded_from_latency_stats():
    # engine with latencies [1.0, 1.0, 0.01, 0.01, 0.01], warmup=2
    # -> p50 near 10ms, not near 1000ms
def test_p95_is_the_95th_percentile_not_the_max():
    # 20 scripted latencies; assert p95 != max
def test_rtf_is_total_inference_time_over_total_audio_duration():
    # 2 samples of 1.0s audio, 0.25s latency each -> rtf == 0.25
def test_benchmark_reports_n_utterances_excluding_warmup():
def test_host_info_includes_thread_count():
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement `harness.py`**

Time with `time.perf_counter` around `transcribe` only, never around audio decode. Warmup runs reuse the first sample and are discarded entirely. Percentiles via `numpy.percentile` with linear interpolation. `host_info` reads `platform.processor()` and `os.cpu_count()`.

- [ ] **Step 4: Run to verify they pass**

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: benchmark harness with warmup exclusion and percentile latency"
```

---

### Task 7: Dataset loading, bench CLI and report

**Files:**
- Create: `src/whisper_cpu_serve/bench/dataset.py`
- Create: `src/whisper_cpu_serve/bench/report.py`
- Create: `scripts/run_bench.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Produces:
  - `load_samples(limit: int | None = None) -> list[Sample]` from `hf-internal-testing/librispeech_asr_dummy`, resampled through `audio` so the benchmark and the API share one preprocessing path
  - `to_markdown(metrics: list[EngineMetrics], host: dict) -> str`
  - `scripts/run_bench.py --engines torch_fp32,onnx_fp32,onnx_int8 --limit N`, writing `bench/results/latest.json` and `RESULTS.md`

- [ ] **Step 1: Write the failing report test**

```python
def test_markdown_table_has_one_row_per_engine_and_records_host():
def test_markdown_reports_speedup_relative_to_first_engine():
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Implement `dataset.py`, `report.py`, `scripts/run_bench.py`**

Decode each dataset row's array through `audio` rather than trusting the dataset's own sample rate.

- [ ] **Step 4: Run to verify it passes**

- [ ] **Step 5: Commit**

---

### Task 8: FastAPI service

**Files:**
- Create: `src/whisper_cpu_serve/api.py`
- Create: `tests/test_api.py`

**Interfaces:**
- Consumes: `Transcriber`, `FakeTranscriber`, `audio`, `config`
- Produces:
  - `create_app(engine: Transcriber) -> FastAPI` — dependency-injectable, so tests never load a model
  - `GET /health` -> `{status, model, engine, thread_count, version}`
  - `POST /transcribe` (multipart `file`) -> `{text, audio_seconds, latency_ms, engine}`

- [ ] **Step 1: Write the failing tests** (all against `FakeTranscriber`, all fast)

```python
def test_health_reports_engine_and_thread_count():
def test_transcribe_returns_text_duration_latency_and_engine():
def test_zero_byte_upload_returns_400():                      # Review Focus 1
def test_text_file_named_wav_returns_415():                   # Review Focus 2
def test_audio_just_over_120s_returns_413():                  # Review Focus 5
def test_audio_exactly_120s_is_accepted():                    # Review Focus 5
def test_missing_file_field_returns_422():
```

- [ ] **Step 2: Run to verify they fail**

- [ ] **Step 3: Implement `api.py`**

Map `EmptyAudioError` -> 400, `AudioDecodeError` -> 415, over-length -> 413. The 120s check uses `>` on `duration_seconds` so exactly 120.0 is accepted. Module-level `app` built from the engine named by `ENGINE` env var, constructed in a `lifespan` handler so the model loads once; `create_app` stays importable without touching a model.

- [ ] **Step 4: Run to verify they pass**

Run: `python -m pytest tests/test_api.py -v`

- [ ] **Step 5: Commit**

---

### Task 9: Container and Hugging Face Space

**Files:**
- Create: `Dockerfile`, `.dockerignore`, `requirements.txt`
- Create: `README.md` (with Space YAML frontmatter)

- [ ] **Step 1: Write `Dockerfile`**

`python:3.11-slim`; no `ffmpeg`. Install `requirements.txt`, copy `src/` and the INT8 model directory only, expose 7860, `CMD uvicorn whisper_cpu_serve.api:app --host 0.0.0.0 --port 7860`. Set `ENGINE=onnx_int8` and `OMP_NUM_THREADS=2` (free Space has 2 vCPU). `.dockerignore` excludes `models/onnx-fp32/`, `tests/`, `docs/`, `bench/results/`.

- [ ] **Step 2: Build and verify the image serves**

```bash
docker build -t whisper-cpu-serve . && docker run -d -p 7860:7860 --name wcs whisper-cpu-serve
curl -s localhost:7860/health
```
Expected: JSON reporting `engine: onnx_int8`. Then `curl -F file=@sample.wav localhost:7860/transcribe` returns a transcript.

- [ ] **Step 3: Write `README.md`**

Space frontmatter (`sdk: docker`, `app_port: 7860`), what the project does, the results table, the exact benchmark command, **the stated eval-set size and host hardware**, and an explicit note that hosted Space latency is not comparable to benchmark latency.

- [ ] **Step 4: Commit**

---

### Task 10: Run the real benchmark and record results

**Files:**
- Modify: `README.md`, create `RESULTS.md`, `bench/results/latest.json`

- [ ] **Step 1: Run the full benchmark**

```bash
python scripts/run_bench.py --engines torch_fp32,onnx_fp32,onnx_int8
```

- [ ] **Step 2: Verify every table cell is a measured number**

No cell may be estimated, copied from documentation, or predicted. If an engine failed to run, its row says so.

- [ ] **Step 3: Record findings, including unfavorable ones**

If INT8 regressed WER more than it improved latency, say exactly that in `RESULTS.md`. That is a legitimate result per the spec and must not be tuned away.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "docs: benchmark results on <cpu>, <n> utterances"
```
