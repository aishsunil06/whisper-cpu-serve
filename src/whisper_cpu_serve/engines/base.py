"""The engine interface.

One method, structurally typed. Every engine -- Torch, ONNX, quantized ONNX, or
a fake -- satisfies this protocol, which is what lets the benchmark swap models
while holding preprocessing, decoding and audio identical. Without that
guarantee a measured latency difference would be unattributable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Transcript:
    """A transcription result, tagged with the engine that produced it."""

    text: str
    engine: str


@runtime_checkable
class Transcriber(Protocol):
    """Anything that turns 16kHz mono float32 audio into text."""

    name: str

    def transcribe(self, audio: np.ndarray) -> Transcript:
        """Transcribe preprocessed audio.

        The caller is responsible for preprocessing (see `audio.load_audio`);
        engines must not decode or resample, or the benchmark would be measuring
        preprocessing differences alongside model differences.
        """
        ...

    def model_size_bytes(self) -> int:
        """On-disk or in-memory size of the model's weights."""
        ...
