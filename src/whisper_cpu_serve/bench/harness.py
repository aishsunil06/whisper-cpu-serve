"""Benchmark harness: drives samples through an engine and measures the result.

Three measurement decisions matter more than the code:

1. Warmup runs are discarded entirely. The first inference pays for lazy graph
   initialization and memory allocation; counting it would attribute startup
   cost to the model and invent a speedup that is not there.

2. Only `transcribe` is timed. Audio decoding happens once, up front, outside
   the timer -- otherwise the benchmark would measure preprocessing that is
   identical across engines and dilute the difference being measured.

3. Latency is reported as percentiles, not a mean. A mean hides the tail, and
   the tail is what a user actually waits for.
"""

from __future__ import annotations

import os
import platform
import time
from dataclasses import dataclass

import numpy as np

from .. import config
from ..engines.base import Transcriber
from ..threads import observed_thread_count
from ..wer import corpus_wer


@dataclass(frozen=True)
class Sample:
    """One evaluation item: preprocessed audio and its ground-truth text."""

    audio: np.ndarray
    reference: str


@dataclass(frozen=True)
class EngineMetrics:
    """Everything measured for one engine over one evaluation set."""

    engine: str
    wer: float
    p50_ms: float
    p95_ms: float
    rtf: float
    model_size_mb: float
    n_utterances: int


def host_info() -> dict:
    """Hardware context. Latency numbers without this are not quotable."""
    return {
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
        "logical_cores": os.cpu_count() or 1,
        "thread_count": observed_thread_count(),
        "thread_count_requested": config.THREAD_COUNT,
    }


def benchmark(
    engine: Transcriber, samples: list[Sample], warmup: int = 2
) -> EngineMetrics:
    """Measure WER, latency percentiles and real-time factor for one engine."""
    if not samples:
        raise ValueError("benchmark requires at least one sample")

    for _ in range(warmup):
        engine.transcribe(samples[0].audio)

    latencies_s: list[float] = []
    hypotheses: list[str] = []
    total_audio_s = 0.0

    for sample in samples:
        start = time.perf_counter()
        transcript = engine.transcribe(sample.audio)
        latencies_s.append(time.perf_counter() - start)

        hypotheses.append(transcript.text)
        total_audio_s += len(sample.audio) / 16000

    latencies_ms = np.array(latencies_s) * 1000.0

    return EngineMetrics(
        engine=engine.name,
        wer=corpus_wer([s.reference for s in samples], hypotheses),
        p50_ms=float(np.percentile(latencies_ms, 50)),
        p95_ms=float(np.percentile(latencies_ms, 95)),
        rtf=sum(latencies_s) / total_audio_s,
        model_size_mb=engine.model_size_bytes() / 1e6,
        n_utterances=len(samples),
    )
