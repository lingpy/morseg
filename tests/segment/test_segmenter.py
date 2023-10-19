from morseg.segment.segmenter import Segmenter


def test_segmenter_interface():
    words = [
        [
            ['r', 'y', 'k', '+', 'ə', 'n', '+', 'f', 'l', 'ɔ', 's', '+', 'ə'],
            ['ʃ', 't', 'a', 'x', '+', 'ə', 'l', '+', 'r', 'ɔ', 'x', '+', 'ə', 'n']
        ]
    ]

    segmenter = Segmenter(words)
    assert segmenter.sequences == words
    assert segmenter.BOUNDARY_SYMBOL == '+'
    assert segmenter.EOS_SYMBOL == '#'

    # test stub for segment_all method to achieve 100% test coverage
    # (method should not be used in practice)
    assert segmenter.segment_all() == words

    # test whether assigning special symbol via kwargs works
    segmenter = Segmenter(words, eos_symbol='$', boundary_symbol='-')
    assert segmenter.BOUNDARY_SYMBOL == '-'
    assert segmenter.EOS_SYMBOL == '$'
