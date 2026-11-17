"""The Torch baseline, against a real clip. Slow: loads and runs a real model."""

import pytest

from whisper_cpu_serve.wer import word_error_rate


@pytest.mark.slow
def test_transcribes_a_known_clip_below_loose_wer_threshold(real_clip):
    """A smoke test that the model works, deliberately not an accuracy claim.

    The threshold is loose on purpose: measuring accuracy is the benchmark's
    job, not this test's. This only catches a model that is wired up wrong.
    """
    from whisper_cpu_serve.engines.torch_whisper import TorchWhisper

    audio, reference = real_clip
    result = TorchWhisper().transcribe(audio)

    assert result.engine == "torch_fp32"
    assert word_error_rate(reference, result.text) < 0.25


@pytest.mark.slow
def test_reports_a_plausible_model_size(real_clip):
    from whisper_cpu_serve.engines.torch_whisper import TorchWhisper

    # whisper-base is 74M params at fp32, so comfortably over 100MB.
    assert TorchWhisper().model_size_bytes() > 100_000_000
