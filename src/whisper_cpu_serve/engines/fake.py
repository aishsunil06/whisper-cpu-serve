"""A Transcriber that loads no model.

Exists so two things can be tested without paying for inference: the API's
request handling, and -- more importantly -- the benchmark harness itself. The
harness produces every number this project quotes, so its warmup exclusion and
percentile maths are verified against scripted latencies rather than against a
real model whose timings cannot be controlled.
"""

from __future__ import annotations

import time

import numpy as np

from .base import Transcript


class FakeTranscriber:
    """Returns fixed text after a scripted delay.

    Pass `latency_s` for a constant delay, or `latencies` for a per-call
    sequence (the harness needs this to prove warmup runs are discarded).
    """

    def __init__(
        self,
        text: str = "the quick brown fox",
        latency_s: float = 0.0,
        latencies: list[float] | None = None,
        name: str = "fake",
        size_bytes: int = 1024,
    ) -> None:
        self.name = name
        self._text = text
        self._latency_s = latency_s
        self._latencies = list(latencies) if latencies else None
        self._call_index = 0
        self._size_bytes = size_bytes

    def transcribe(self, audio: np.ndarray) -> Transcript:
        if self._latencies:
            delay = self._latencies[min(self._call_index, len(self._latencies) - 1)]
        else:
            delay = self._latency_s
        self._call_index += 1

        if delay:
            time.sleep(delay)
        return Transcript(text=self._text, engine=self.name)

    def model_size_bytes(self) -> int:
        return self._size_bytes
