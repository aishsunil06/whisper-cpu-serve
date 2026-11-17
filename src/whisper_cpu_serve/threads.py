"""Thread configuration, shared by every engine.

This lives in one place because the benchmark compares engines on latency: if
one engine configured threading and another inherited the machine default, the
measured difference would partly be a difference in parallelism rather than in
the model.

Note that ONNX Runtime's intra-op threads cover only the ONNX session. Whisper's
feature extraction and optimum's generation loop run in torch, so torch's thread
count matters for the ONNX engines too.
"""

from __future__ import annotations

import torch

from . import config


def configure_threads() -> None:
    """Pin torch to the configured thread count. Safe to call repeatedly."""
    torch.set_num_threads(config.THREAD_COUNT)


def observed_thread_count() -> int:
    """The thread count actually in effect, not the one requested.

    `/health` and the benchmark report this rather than `config.THREAD_COUNT`,
    so a mismatch between intent and reality is visible instead of hidden.
    """
    return torch.get_num_threads()
