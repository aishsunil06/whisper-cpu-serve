"""PyTorch fp32 baseline.

This is the engine every optimization is measured against. It is deliberately
the most ordinary possible implementation -- stock transformers, no tricks --
because a baseline that has been tuned makes the speedup numbers meaningless.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .. import config
from ..audio import TARGET_SAMPLE_RATE
from ..threads import configure_threads
from .base import Transcript


class TorchWhisper:
    """Whisper inference through stock PyTorch on CPU."""

    def __init__(self, model_id: str = config.MODEL_ID) -> None:
        self.name = "torch_fp32"
        self.model_id = model_id

        configure_threads()
        self._processor = WhisperProcessor.from_pretrained(model_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(model_id)
        self._model.eval()

    def transcribe(self, audio: np.ndarray) -> Transcript:
        features = self._processor(
            audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        ).input_features

        with torch.inference_mode():
            token_ids = self._model.generate(features, language="en", task="transcribe")

        text = self._processor.batch_decode(token_ids, skip_special_tokens=True)[0]
        return Transcript(text=text.strip(), engine=self.name)

    def model_size_bytes(self) -> int:
        return sum(p.numel() * p.element_size() for p in self._model.parameters())
