import pytest
from morseg.segment.kirschenbaum import KirschenbaumMorphemeSegmenter, Scorer


@pytest.fixture
def words(test_data):
    words = [
        [
            ['ts', 'au', 'b', 'ə', 'r', 'ə', 'r'],
            ['ts', 'au', 'b', 'ə', 'r'],
            ['ts', 'au', 'b', 'ə', 'r', 'ai']
        ],
        [
            ['f', 'ɪ', 'ʃ'],
            ['h', 'ai', 'f', 'ɪ', 'ʃ'],
            ['f', 'ɪ', 'ʃ', 'ə', 'r', 'n', 'ɛ', 'ts'],
            ['f', 'ɪ', 'ʃ', 'g', 'ɪ', 'f', 't'],
            ['f', 'ɪ', 'ʃ', 'ə', 'r']
        ],
        [
            ['f', 'ɪ', 'ʃ', 'g', 'ɪ', 'f', 't'],
            ['f', 'ɪ', 'ʃ', 'ə', 'r']
        ]
    ]

    return words


@pytest.fixture
def kirschenbaum_segmenter_a(words):
    return KirschenbaumMorphemeSegmenter(words)


@pytest.fixture
def kirschenbaum_segmenter_b(words):
    return KirschenbaumMorphemeSegmenter(words, scoring_scheme="B")


# TODO more in-depth testing for pattern finding and different scoring schemes
def test_segmentation_a(kirschenbaum_segmenter_a):
    assert kirschenbaum_segmenter_a.get_best_segmentation(['h', 'ai', 'f', 'ɪ', 'ʃ']) == ['h', 'ai', '+', 'f', 'ɪ', 'ʃ']
    assert kirschenbaum_segmenter_a.get_best_segmentation(['ts', 'au', 'b', 'ə', 'r', 'ai']) == ['ts', 'au', 'b', '+', 'ə', 'r', 'ai']

    assert (kirschenbaum_segmenter_a.get_best_segmentation(['h', 'ai', 'f', 'ɪ', 'ʃ'], score_based=False)
            == ['h', 'ai', '+', 'f', 'ɪ', 'ʃ'])
    assert (kirschenbaum_segmenter_a.get_best_segmentation(['ts', 'au', 'b', 'ə', 'r', 'ai'], score_based=False)
            == ['ts', 'au', 'b', '+', 'ə', 'r', 'ai'])


def test_segmentation_b(kirschenbaum_segmenter_b):
    assert kirschenbaum_segmenter_b.get_best_segmentation(['h', 'ai', 'f', 'ɪ', 'ʃ']) == ['h', 'ai', '+', 'f', 'ɪ', 'ʃ']
    assert kirschenbaum_segmenter_b.get_best_segmentation(['ts', 'au', 'b', 'ə', 'r', 'ai']) == ['ts', 'au', 'b', '+', 'ə', 'r', 'ai']

