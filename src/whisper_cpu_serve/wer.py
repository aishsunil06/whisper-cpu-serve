"""Normalized word error rate.

Two decisions here carry the whole evaluation:

1. Normalization (lowercase, strip punctuation, collapse whitespace). Whisper
   emits "The cat, sat." where LibriSpeech's reference is "the cat sat". Without
   normalization the measured WER is dominated by formatting and the comparison
   between engines becomes noise.

2. Corpus-level aggregation over total words, not the mean of per-utterance
   rates. Averaging rates weights a one-word utterance as heavily as a
   thirty-word one, which systematically misreports accuracy on sets containing
   short clips.
"""

from __future__ import annotations

import re
import string

import jiwer

_PUNCTUATION = str.maketrans("", "", string.punctuation)
_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    return _WHITESPACE.sub(" ", text.lower().translate(_PUNCTUATION)).strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER for a single utterance, after normalization.

    An empty reference has no words to divide by, so it is defined explicitly
    rather than left to raise: a perfect match is 0.0, anything else is 1.0.
    """
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return float(jiwer.wer(ref, hyp))


def corpus_wer(references: list[str], hypotheses: list[str]) -> float:
    """WER aggregated over all words in the corpus.

    Total edit distance over total reference words, so long utterances carry
    proportionally more weight than short ones.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"got {len(references)} references and {len(hypotheses)} hypotheses"
        )

    refs = [normalize(r) for r in references]
    hyps = [normalize(h) for h in hypotheses]

    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    if not pairs:
        return 0.0 if not any(hyps) else 1.0

    output = jiwer.process_words([r for r, _ in pairs], [h for _, h in pairs])
    errors = output.substitutions + output.deletions + output.insertions
    total_words = sum(len(r.split()) for r, _ in pairs)
    return errors / total_words
