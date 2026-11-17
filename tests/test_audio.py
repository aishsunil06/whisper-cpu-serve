"""Audio preprocessing: everything downstream assumes 16kHz mono float32."""

import io

import numpy as np
import pytest
import soundfile as sf

from whisper_cpu_serve.audio import (
    AudioDecodeError,
    EmptyAudioError,
    chunk,
    duration_seconds,
    load_audio,
)

TARGET_SR = 16000


def _wav_bytes(samples: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _sine(seconds: float, sample_rate: int, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(seconds * sample_rate), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_resamples_8k_to_16k_doubling_sample_count():
    data = _wav_bytes(_sine(1.0, 8000), 8000)
    assert len(load_audio(data)) == pytest.approx(TARGET_SR, abs=2)


def test_resamples_44100_to_16k():
    data = _wav_bytes(_sine(1.0, 44100), 44100)
    assert len(load_audio(data)) == pytest.approx(TARGET_SR, abs=2)


def test_native_16k_is_left_alone():
    data = _wav_bytes(_sine(1.0, TARGET_SR), TARGET_SR)
    assert len(load_audio(data)) == TARGET_SR


def test_stereo_collapses_to_mono():
    mono = _sine(1.0, TARGET_SR)
    stereo = np.stack([mono, mono], axis=1)
    audio = load_audio(_wav_bytes(stereo, TARGET_SR))
    assert audio.ndim == 1
    assert len(audio) == TARGET_SR


def test_returns_float32_in_unit_range():
    audio = load_audio(_wav_bytes(_sine(1.0, TARGET_SR), TARGET_SR))
    assert audio.dtype == np.float32
    assert np.abs(audio).max() <= 1.0


def test_duration_seconds_of_16000_samples_is_1():
    assert duration_seconds(np.zeros(TARGET_SR, dtype=np.float32)) == 1.0


def test_chunk_125s_into_30s_windows_with_1s_overlap_yields_5_chunks():
    audio = np.zeros(125 * TARGET_SR, dtype=np.float32)
    assert len(chunk(audio, window_s=30.0, overlap_s=1.0)) == 5


def test_chunk_shorter_than_window_returns_single_chunk():
    audio = np.zeros(5 * TARGET_SR, dtype=np.float32)
    chunks = chunk(audio, window_s=30.0, overlap_s=1.0)
    assert len(chunks) == 1
    assert len(chunks[0]) == len(audio)


def test_empty_audio_raises_EmptyAudioError():
    with pytest.raises(EmptyAudioError):
        load_audio(b"")


def test_non_audio_bytes_raise_AudioDecodeError():
    with pytest.raises(AudioDecodeError):
        load_audio(b"this is a text file pretending to be audio" * 10)
