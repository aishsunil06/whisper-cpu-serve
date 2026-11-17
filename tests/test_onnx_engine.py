"""ONNX engines. Slow: loads real models exported by scripts/export_onnx.py."""

import pytest

from whisper_cpu_serve import config
from whisper_cpu_serve.wer import word_error_rate

pytestmark = pytest.mark.skipif(
    not (config.ONNX_INT8_DIR / "encoder_model.onnx").exists(),
    reason="run scripts/export_onnx.py first",
)


@pytest.mark.slow
def test_fp32_onnx_transcribes_a_known_clip(real_clip):
    from whisper_cpu_serve.engines.onnx_whisper import OnnxWhisper

    audio, reference = real_clip
    result = OnnxWhisper(config.ONNX_FP32_DIR).transcribe(audio)
    assert result.engine == "onnx_fp32"
    assert word_error_rate(reference, result.text) < 0.25


@pytest.mark.slow
def test_int8_onnx_transcribes_a_known_clip(real_clip):
    """Quantization must not destroy the model. Accuracy cost is the
    benchmark's business; this only catches a broken quantized graph."""
    from whisper_cpu_serve.engines.onnx_whisper import OnnxWhisper

    audio, reference = real_clip
    result = OnnxWhisper(config.ONNX_INT8_DIR, quantized=True).transcribe(audio)
    assert result.engine == "onnx_int8"
    assert word_error_rate(reference, result.text) < 0.25


@pytest.mark.slow
def test_int8_is_materially_smaller_than_fp32():
    from whisper_cpu_serve.engines.onnx_whisper import OnnxWhisper

    fp32 = OnnxWhisper(config.ONNX_FP32_DIR).model_size_bytes()
    int8 = OnnxWhisper(config.ONNX_INT8_DIR, quantized=True).model_size_bytes()
    assert int8 < fp32 / 2
