"""Audio ingest and preprocessing.

Everything downstream of this module assumes 16kHz mono float32. The benchmark
and the API both decode through here, so a latency comparison between engines
cannot be contaminated by differing preprocessing.

Decoding is libsndfile (via soundfile) only -- deliberately no ffmpeg, which
keeps the container lean. libsndfile handles WAV and FLAC natively, which covers
LibriSpeech and ordinary uploads.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

TARGET_SAMPLE_RATE = 16000


class AudioError(Exception):
    """Base class for audio input problems."""


class EmptyAudioError(AudioError):
    """Input contained no audio samples."""


class AudioDecodeError(AudioError):
    """Input could not be decoded as audio."""


def load_audio(data: bytes | str | Path) -> np.ndarray:
    """Decode audio to 16kHz mono float32.

    Accepts raw bytes (an upload) or a filesystem path.
    """
    if isinstance(data, bytes):
        if len(data) == 0:
            raise EmptyAudioError("received zero bytes")
        source: io.BytesIO | str = io.BytesIO(data)
    else:
        source = str(data)

    try:
        samples, sample_rate = sf.read(source, dtype="float32", always_2d=False)
    except Exception as exc:  # libsndfile raises several unrelated types
        raise AudioDecodeError(f"could not decode audio: {exc}") from exc

    if samples.size == 0:
        raise EmptyAudioError("decoded audio contained no samples")

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    if sample_rate != TARGET_SAMPLE_RATE:
        samples = soxr.resample(samples, sample_rate, TARGET_SAMPLE_RATE)

    return np.ascontiguousarray(samples, dtype=np.float32)


def duration_seconds(audio: np.ndarray) -> float:
    """Length of 16kHz audio in seconds."""
    return len(audio) / TARGET_SAMPLE_RATE


def chunk(
    audio: np.ndarray, window_s: float = 30.0, overlap_s: float = 1.0
) -> list[np.ndarray]:
    """Split audio into overlapping windows.

    Whisper's receptive field is 30s; longer audio must be windowed. The overlap
    exists so a word straddling a boundary appears whole in at least one window.
    """
    if overlap_s >= window_s:
        raise ValueError("overlap_s must be smaller than window_s")

    window = int(window_s * TARGET_SAMPLE_RATE)
    stride = int((window_s - overlap_s) * TARGET_SAMPLE_RATE)

    if len(audio) <= window:
        return [audio]

    chunks = []
    for start in range(0, len(audio), stride):
        chunks.append(audio[start : start + window])
        if start + window >= len(audio):
            break
    return chunks
