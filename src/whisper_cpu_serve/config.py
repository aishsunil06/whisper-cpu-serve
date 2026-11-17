"""Project-wide constants.

Model id and thread count live here because they must be identical between the
benchmark and the served API. A benchmark run with a different thread count than
production measures a machine that does not exist.
"""

from __future__ import annotations

import os
from pathlib import Path

# whisper-base (74M), not whisper-small (244M). On CPU, running three engines
# over the eval set with `small` makes a full benchmark run tens of minutes,
# which is too slow to iterate on inside this project's timebox. Every part of
# the comparison holds at `base`; raise this constant to re-run at `small`.
MODEL_ID = os.environ.get("MODEL_ID", "openai/whisper-base")

# CPU inference latency is meaningless without a recorded thread count.
THREAD_COUNT = int(os.environ.get("OMP_NUM_THREADS", "4"))

MAX_AUDIO_SECONDS = 120.0

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
ONNX_FP32_DIR = MODELS_DIR / "onnx-fp32"
ONNX_INT8_DIR = MODELS_DIR / "onnx-int8"

EVAL_DATASET = "hf-internal-testing/librispeech_asr_dummy"
