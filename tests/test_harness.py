"""The benchmark harness.

Every number this project quotes comes out of here, so the harness is tested
against a fake engine with scripted latencies rather than a real model. An
untested measurement tool fails silently -- you would never know the speedup was
an artifact of counting warmup runs.
"""

import numpy as np
import pytest

from whisper_cpu_serve.bench.harness import Sample, benchmark, host_info
from whisper_cpu_serve.engines.fake import FakeTranscriber

ONE_SECOND = np.zeros(16000, dtype=np.float32)


def _samples(n: int, reference: str = "the quick brown fox") -> list[Sample]:
    return [Sample(audio=ONE_SECOND, reference=reference) for _ in range(n)]


def test_warmup_runs_are_excluded_from_latency_stats():
    """Two slow warmups then three fast runs must report fast latency.

    Including warmup is the single most likely way to accidentally fabricate a
    speedup, because lazy initialization lands entirely on the first call.
    """
    engine = FakeTranscriber(latencies=[0.20, 0.20, 0.01, 0.01, 0.01])
    metrics = benchmark(engine, _samples(3), warmup=2)
    assert metrics.p50_ms < 100


def test_warmup_runs_are_not_counted_as_utterances():
    engine = FakeTranscriber()
    assert benchmark(engine, _samples(3), warmup=2).n_utterances == 3


def test_p95_is_a_percentile_not_the_maximum():
    latencies = [0.001] * 19 + [0.5]
    engine = FakeTranscriber(latencies=latencies)
    metrics = benchmark(engine, _samples(20), warmup=0)
    assert metrics.p95_ms < 500


def test_rtf_is_inference_time_over_audio_duration():
    """Two 1-second clips at ~0.05s each is a real-time factor near 0.05."""
    engine = FakeTranscriber(latency_s=0.05)
    metrics = benchmark(engine, _samples(2), warmup=0)
    assert metrics.rtf == pytest.approx(0.05, abs=0.04)


def test_wer_is_computed_against_references():
    engine = FakeTranscriber(text="the quick brown fox")
    assert benchmark(engine, _samples(2), warmup=0).wer == 0.0


def test_wrong_transcription_produces_nonzero_wer():
    engine = FakeTranscriber(text="completely different words here")
    assert benchmark(engine, _samples(2), warmup=0).wer > 0.0


def test_metrics_carry_engine_name_and_model_size():
    engine = FakeTranscriber(name="fake", size_bytes=2048)
    metrics = benchmark(engine, _samples(1), warmup=0)
    assert metrics.engine == "fake"
    assert metrics.model_size_mb == pytest.approx(2048 / 1e6)


def test_benchmark_requires_at_least_one_sample():
    with pytest.raises(ValueError):
        benchmark(FakeTranscriber(), [], warmup=0)


def test_host_info_records_thread_count_and_cpu():
    info = host_info()
    assert info["thread_count"] >= 1
    assert "cpu" in info
    assert info["logical_cores"] >= 1


def test_rtf_uses_audio_duration_not_sample_count():
    """Mixed durations: RTF and mean latency must diverge.

    With every sample the same length, sum(latency)/total_audio and
    sum(latency)/n are numerically identical, so a wrong implementation passes.
    One 1s clip plus one 3s clip separates them: 4s of audio at 0.05s per call
    is 0.1/4 = 0.025, whereas a per-sample mean would be 0.05.
    """
    engine = FakeTranscriber(latency_s=0.05)
    samples = [
        Sample(audio=np.zeros(16000, dtype=np.float32), reference="a"),
        Sample(audio=np.zeros(48000, dtype=np.float32), reference="a"),
    ]
    metrics = benchmark(engine, samples, warmup=0)
    assert metrics.rtf == pytest.approx(0.025, abs=0.012)
