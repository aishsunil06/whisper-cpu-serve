"""Long-audio transcription.

The API accepts up to 120s but Whisper's receptive field is 30s. Without
windowing, the feature extractor silently pads/truncates to exactly 30s and
everything past that is discarded with no error and no signal to the caller --
the worst class of bug, because the response still looks correct.
"""

import numpy as np
import pytest

from whisper_cpu_serve.engines.fake import FakeTranscriber
from whisper_cpu_serve.transcribe import transcribe_audio

SR = 16000


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_short_audio_is_transcribed_in_one_pass():
    engine = FakeTranscriber(text="hello")
    result = transcribe_audio(engine, _silence(5.0))
    assert result.text == "hello"
    assert result.n_chunks == 1


def test_audio_at_exactly_30s_is_still_one_pass():
    engine = FakeTranscriber(text="hello")
    assert transcribe_audio(engine, _silence(30.0)).n_chunks == 1


def test_90s_audio_is_split_into_multiple_windows():
    """The bug this module exists to prevent: 90s must not become 30s."""
    engine = FakeTranscriber(text="part")
    result = transcribe_audio(engine, _silence(90.0))
    assert result.n_chunks == 3


def test_every_window_is_transcribed_and_joined():
    engine = FakeTranscriber(text="part")
    result = transcribe_audio(engine, _silence(90.0))
    assert result.text == "part part part"


def test_no_window_exceeds_the_models_receptive_field():
    """A window longer than 30s would itself be silently truncated."""
    captured = []

    class Recorder:
        name = "recorder"

        def transcribe(self, audio):
            captured.append(len(audio))
            from whisper_cpu_serve.engines.base import Transcript

            return Transcript(text="x", engine=self.name)

        def model_size_bytes(self) -> int:
            return 1

    transcribe_audio(Recorder(), _silence(100.0))
    assert captured
    assert max(captured) <= 30 * SR


def test_windows_cover_the_entire_input():
    """Total windowed samples must account for all the audio, not a prefix."""
    captured = []

    class Recorder:
        name = "recorder"

        def transcribe(self, audio):
            captured.append(len(audio))
            from whisper_cpu_serve.engines.base import Transcript

            return Transcript(text="x", engine=self.name)

        def model_size_bytes(self) -> int:
            return 1

    transcribe_audio(Recorder(), _silence(100.0))
    assert sum(captured) == 100 * SR


def test_result_reports_the_engine_name():
    assert transcribe_audio(FakeTranscriber(), _silence(5.0)).engine == "fake"


def test_blank_window_transcriptions_are_not_joined_as_empty_gaps():
    engine = FakeTranscriber(text="")
    result = transcribe_audio(engine, _silence(90.0))
    assert result.text == ""
