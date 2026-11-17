"""Shared fixtures. The real-clip fixture downloads ~9MB on first use."""

import pytest


@pytest.fixture(scope="session")
def real_clip():
    """One LibriSpeech clip as (audio_16k_mono_float32, reference_text).

    The audio column is loaded undecoded and passed through the project's own
    loader, so tests and the benchmark see byte-identical preprocessing.
    """
    from datasets import Audio, load_dataset

    from whisper_cpu_serve import config
    from whisper_cpu_serve.audio import load_audio

    ds = load_dataset(config.EVAL_DATASET, "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    row = ds[0]
    return load_audio(row["audio"]["bytes"]), row["text"]
