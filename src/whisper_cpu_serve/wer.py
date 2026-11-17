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

    # jiwer raises on an empty reference, so those pairs are scored by hand
    # rather than dropped. Dropping them would remove each inserted word from
    # the numerator as well as the denominator, scoring a hallucinated
    # transcript against silence as a perfect result.
    scorable = [(r, h) for r, h in zip(refs, hyps) if r]
    empty_reference_insertions = sum(
        len(h.split()) for r, h in zip(refs, hyps) if not r
    )

    total_words = sum(len(r.split()) for r, _ in scorable)
    if total_words == 0:
        raise ValueError(
            "cannot compute corpus WER: references contain no words "
            f"(got {len(references)} references, all empty)"
        )

    errors = empty_reference_insertions
    if scorable:
        output = jiwer.process_words(
            [r for r, _ in scorable], [h for _, h in scorable]
        )
        errors += output.substitutions + output.deletions + output.insertions

    return errors / total_words
