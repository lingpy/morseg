import pytest

from morseg.utils.eval import f_score


def test_f_score():
    reference_words = [
        ['r', 'y', 'k', '+', 'ə', 'n', '+', 'f', 'l', 'ɔ', 's', '+', 'ə'],
        ['ʃ', 't', 'a', 'x', '+', 'ə', 'l', '+', 'r', 'ɔ', 'x', '+', 'ə', 'n']
    ]

    predicted_words = reference_words

    # let's start with perfect predictions, all values should be 1
    f1, precision, recall = f_score(reference_words, predicted_words)
    assert f1 == precision == recall == 1

    predicted_words = [
        ['r', 'y', 'k', 'ə', 'n', '+', 'f', 'l', 'ɔ', 's', '+', 'ə'],
        ['ʃ', 't', 'a', 'x', '+', 'ə', 'l', '+', 'r', 'ɔ', 'x', 'ə', 'n']
    ]

    # this case should have perfect precision, but a lower recall (and thus f1-score)
    f1, precision, recall = f_score(reference_words, predicted_words)
    assert precision == 1
    assert recall == 2 / 3
    assert f1 == 0.8

    predicted_words = [
        ['r', '+', 'y', 'k', '+', 'ə', 'n', '+', 'f', '+', 'l', 'ɔ', 's', '+', 'ə'],
        ['ʃ', 't', 'a', '+', 'x', '+', 'ə', 'l', '+', 'r', 'ɔ', 'x', '+', 'ə', 'n']
    ]

    # now we should have perfect recall, but a lower precision
    f1, precision, recall = f_score(reference_words, predicted_words)
    assert recall == 1
    assert precision == 2 / 3
    assert f1 == 0.8

    predicted_words = [
        ['r', 'y', '+', 'k', 'ə', 'n', 'f', '+', 'l', 'ɔ', 's', 'ə', '+'],
        ['+', 'ʃ', 't', 'a', '+', 'x', 'ə', 'l', 'r', '+', 'ɔ', 'x', 'ə', 'n']
    ]

    # all boundaries are at a wrong position, should yield an f-score of 0.
    # checks also if boundary symbols at the edges can be processed (even though they should not be there)
    f1, precision, recall = f_score(reference_words, predicted_words)
    assert f1 == precision == recall == 0

    # check whether errors are raised correctly
    predicted_words = [
        ['m', 'y', 'k', 'ə', 'n', '+', 'f', 'l', 'ɔ', 's', '+', 'ə'],
        ['ʃ', 't', 'a', 'x', '+', 'ə', 'l', '+', 'r', 'ɔ', 'x', 'ə', 'n']
    ]

    # error should be raised if the (non-boundary) segments of two compared words are not equal.
    try:
        _ = f_score(reference_words, predicted_words)
        pytest.fail()
    except ValueError as ve:
        assert str(ve) == (f"Mismatch between forms {reference_words[0]} and {predicted_words[0]}: "
                           "All segments other than the boundary symbol need to be identical.")

    predicted_words = [
        ['r', 'y', 'k', '+', 'ə', 'n', '+', 'f', 'l', 'ɔ', 's', '+', 'ə'],
        ['ʃ', 't', 'a', 'x', '+', 'ə', 'l', '+', 'r', 'ɔ', 'x', '+', 'ə', 'n'],
        ['f', 'l', 'ʊ', 's', '+', 'aː', 'l']
    ]

    # error should be raised if the number of predicted words is not equal to the number of reference words
    try:
        _ = f_score(reference_words, predicted_words)
        pytest.fail()
    except ValueError as ve:
        assert str(ve) == ("Dimension mismatch: Lists containing the predicted and reference segmentations"
                           " should be of same length, actually: 3 predicted words, 2 reference words")

    predicted_words = [
        "r y k + ə n + f l ɔ s + ə",
        "ʃ t a x + ə l + r ɔ x + ə n"
    ]

    # error should be raised if input is not a nested list
    try:
        _ = f_score(reference_words, predicted_words)
        pytest.fail()
    except ValueError as ve:
        assert str(ve) == ("Type mismatch: Words should be represented as lists of segments, actually:"
                           f"\n    reference: {reference_words[0]} <class 'list'>"
                           f"\n    predicted: {predicted_words[0]} <class 'str'>")

    # error should be raised if one of the input values is not a list at all
    try:
        _ = f_score(reference_words, "dings")
        pytest.fail()
    except ValueError as ve:
        assert str(ve) == ("Type mismatch: Both reference and predicted words should be passed as lists, actually:"
                           "\n    reference: <class 'list'>\n    predicted: <class 'str'>")
