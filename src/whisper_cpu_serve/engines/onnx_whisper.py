"""ONNX Runtime engines, fp32 and INT8-quantized.

Both variants are the same class: the only difference is which directory of
exported graphs gets loaded. Keeping them one implementation is deliberate --
if fp32 and INT8 had separate code paths, a latency difference between them
could be a difference in the wrapper rather than in the model, which is exactly
the confound the Transcriber protocol exists to rule out.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime
import torch
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import WhisperProcessor

from .. import config
from ..audio import TARGET_SAMPLE_RATE
from ..threads import configure_threads
from .base import Transcript


class OnnxWhisper:
    """Whisper inference through ONNX Runtime on CPU."""

    def __init__(self, model_dir: Path, quantized: bool = False) -> None:
        self.name = "onnx_int8" if quantized else "onnx_fp32"
        self.model_dir = Path(model_dir)

        # ONNX session threads cover the session only; feature extraction and
        # optimum's generation loop run in torch, so both must be pinned.
        configure_threads()

        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = config.THREAD_COUNT

        self._processor = WhisperProcessor.from_pretrained(self.model_dir)
        self._model = ORTModelForSpeechSeq2Seq.from_pretrained(
            self.model_dir,
            session_options=session_options,
            provider="CPUExecutionProvider",
        )

    def transcribe(self, audio: np.ndarray) -> Transcript:
        features = self._processor(
            audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        ).input_features

        with torch.inference_mode():
            token_ids = self._model.generate(
                features, language="en", task="transcribe"
            )
        text = self._processor.batch_decode(token_ids, skip_special_tokens=True)[0]
        return Transcript(text=text.strip(), engine=self.name)

    def model_size_bytes(self) -> int:
        """Total size of the exported graphs on disk.

        Measured from the files rather than from parameter counts, because
        quantization's effect on a served artifact is a file-size question.
        """
        return sum(p.stat().st_size for p in self.model_dir.glob("*.onnx"))
