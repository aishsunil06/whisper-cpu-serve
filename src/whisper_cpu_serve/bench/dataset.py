"""Evaluation set loading.

The audio column is loaded *undecoded* and pushed through the project's own
`load_audio`. Two reasons: it avoids a torchcodec/ffmpeg dependency, and more
importantly it guarantees the benchmark sees byte-identical preprocessing to the
served API. If the dataset decoded its own audio, the benchmark would be
measuring a pipeline that production never runs.
"""

from __future__ import annotations

from .. import config
from ..audio import load_audio
from .harness import Sample


def load_samples(limit: int | None = None) -> list[Sample]:
    """Load the evaluation set as preprocessed Samples.

    Downloads roughly 9MB on first call, then reads from the HF cache.
    """
    from datasets import Audio, load_dataset

    ds = load_dataset(config.EVAL_DATASET, "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    return [
        Sample(audio=load_audio(row["audio"]["bytes"]), reference=row["text"])
        for row in ds
    ]
