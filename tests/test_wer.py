"""Word error rate.

The normalization rules here are the difference between a meaningful number and
a meaningless one: Whisper emits punctuation and casing, LibriSpeech references
have neither, so unnormalized WER measures formatting rather than accuracy.
"""

import pytest

from whisper_cpu_serve.wer import corpus_wer, normalize, word_error_rate


def test_normalize_lowercases_and_strips_punctuation():
    assert normalize("The cat, sat!") == "the cat sat"


def test_normalize_collapses_whitespace():
    assert normalize("the   cat\n sat") == "the cat sat"


def test_punctuation_and_case_differences_give_zero_wer():
    assert word_error_rate("the cat sat", "The cat, sat.") == 0.0


def test_single_substitution_in_three_words_is_one_third():
    assert word_error_rate("the cat sat", "the dog sat") == pytest.approx(1 / 3)


def test_deletion_counts_as_an_error():
    assert word_error_rate("the cat sat", "the sat") == pytest.approx(1 / 3)


def test_empty_reference_with_empty_hypothesis_is_zero():
    assert word_error_rate("", "") == 0.0


def test_empty_reference_with_nonempty_hypothesis_is_one():
    assert word_error_rate("", "hello") == 1.0


def test_corpus_wer_aggregates_over_total_words_not_mean_of_rates():
    """A mean of per-utterance rates would give 0.5 here and overstate accuracy.

    One error across five total words is 0.2. Averaging 0.0 and 1.0 is not.
    """
    refs = ["a b c d", "e"]
    hyps = ["a b c d", "x"]
    assert corpus_wer(refs, hyps) == pytest.approx(0.2)


def test_corpus_wer_of_perfect_transcription_is_zero():
    assert corpus_wer(["the cat sat"], ["The cat, sat."]) == 0.0


def test_corpus_wer_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        corpus_wer(["a", "b"], ["a"])


def test_corpus_wer_counts_insertions_against_an_empty_reference():
    """A hallucinating engine must not be rewarded with silence.

    Dropping empty-reference pairs entirely removes the inserted words from the
    numerator as well as the denominator, scoring three spurious words as a
    perfect transcription.
    """
    assert corpus_wer(["", "a b"], ["spurious words here", "a b"]) > 0.0


def test_corpus_wer_insertion_count_is_exact():
    # 3 insertions over 2 reference words.
    assert corpus_wer(["", "a b"], ["x y z", "a b"]) == pytest.approx(1.5)


def test_corpus_wer_raises_when_every_reference_is_empty():
    """A WER over a zero-word denominator is not a number."""
    with pytest.raises(ValueError):
        corpus_wer(["", ""], ["x y z", "q"])
