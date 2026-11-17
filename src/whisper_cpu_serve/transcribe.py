"""Transcription of audio of any supported length.

Whisper's feature extractor pads or truncates every input to exactly 30 seconds.
Handing it 90 seconds returns a confident transcript of the first 30 and
discards the rest with no error -- so anything above the receptive field must be
windowed here, before it reaches an engine.

Windows are non-overlapping. Overlap would transcribe boundary words twice and
require de-duplication to join cleanly; the cost of not overlapping is that a
word straddling a boundary may be split. That is a visible, bounded error,
whereas duplicated text is a silently wrong transcript.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import TARGET_SAMPLE_RATE, chunk
from .engines.base import Transcriber

# Whisper's receptive field. A window above this is truncated by the extractor.
WINDOW_SECONDS = 30.0


@dataclass(frozen=True)
class LongTranscript:
    """A transcript plus how many windows produced it."""

    text: str
    engine: str
    n_chunks: int


def transcribe_audio(engine: Transcriber, audio: np.ndarray) -> LongTranscript:
    """Transcribe audio of any length, windowing when it exceeds 30 seconds."""
    if len(audio) <= WINDOW_SECONDS * TARGET_SAMPLE_RATE:
        result = engine.transcribe(audio)
        return LongTranscript(text=result.text, engine=result.engine, n_chunks=1)

    windows = chunk(audio, window_s=WINDOW_SECONDS, overlap_s=0.0)
    parts = [engine.transcribe(window) for window in windows]

    text = " ".join(part.text.strip() for part in parts if part.text.strip())
    return LongTranscript(
        text=text,
        engine=parts[0].engine,
        n_chunks=len(windows),
    )
