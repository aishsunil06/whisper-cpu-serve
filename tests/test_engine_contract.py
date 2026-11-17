"""Contract every engine must satisfy.

This is the load-bearing test of the project. The benchmark's claim -- that a
latency difference between engines is a property of the model rather than an
accident of plumbing -- only holds if the engines are genuinely interchangeable.
These assertions run against every implementation, real and fake.

Real engines are appended to ENGINES as they are implemented; the slow ones are
skipped unless -m slow is passed.
"""

import time

import numpy as np
import pytest

from whisper_cpu_serve.engines.base import Transcript
from whisper_cpu_serve.engines.fake import FakeTranscriber

ONE_SECOND_OF_SILENCE = np.zeros(16000, dtype=np.float32)

FAST_ENGINES = [FakeTranscriber]


@pytest.mark.parametrize("factory", FAST_ENGINES)
def test_transcribe_returns_a_transcript(factory):
    result = factory().transcribe(ONE_SECOND_OF_SILENCE)
    assert isinstance(result, Transcript)


@pytest.mark.parametrize("factory", FAST_ENGINES)
def test_transcript_carries_text_and_engine_name(factory):
    engine = factory()
    result = engine.transcribe(ONE_SECOND_OF_SILENCE)
    assert isinstance(result.text, str)
    assert result.engine == engine.name


@pytest.mark.parametrize("factory", FAST_ENGINES)
def test_engine_exposes_a_nonempty_name(factory):
    assert factory().name


@pytest.mark.parametrize("factory", FAST_ENGINES)
def test_model_size_bytes_is_positive(factory):
    assert factory().model_size_bytes() > 0


def test_fake_transcriber_returns_its_scripted_text():
    assert FakeTranscriber(text="hello world").transcribe(
        ONE_SECOND_OF_SILENCE
    ).text == "hello world"


def test_fake_transcriber_respects_scripted_latency():
    engine = FakeTranscriber(latency_s=0.05)
    start = time.perf_counter()
    engine.transcribe(ONE_SECOND_OF_SILENCE)
    assert time.perf_counter() - start >= 0.05


def test_fake_transcriber_cycles_through_scripted_latencies():
    """The harness needs per-call latency control to test warmup exclusion."""
    engine = FakeTranscriber(latencies=[0.0, 0.02])
    first = time.perf_counter()
    engine.transcribe(ONE_SECOND_OF_SILENCE)
    mid = time.perf_counter()
    engine.transcribe(ONE_SECOND_OF_SILENCE)
    end = time.perf_counter()
    assert (mid - first) < 0.02 <= (end - mid)
