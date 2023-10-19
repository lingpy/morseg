import pytest
from morseg.segment.benden import BendenMorphemeSegmenter
from lingpy import Wordlist


@pytest.fixture
def words(test_data):
    wl = Wordlist(str(test_data / "German.tsv"))
    return [wl[idx, "tokens"] for idx in wl]


@pytest.fixture
def benden_segmenter(words):
    return BendenMorphemeSegmenter([words])


# only test one global segmenter for now
def test_init(benden_segmenter):
    assert len(benden_segmenter.tries) == 1
    assert benden_segmenter.tries[0] == benden_segmenter.global_trie


def test_alg1(benden_segmenter):
    test_word = ['yː', 'b', 'ə', 'r', 'm', 'ɔ', 'r', 'g', 'ə', 'n']
    ref = ['yː', 'b', 'ə', '+', 'r', '+', 'm', 'ɔ', 'r', 'g', 'ə', 'n']
    assert benden_segmenter.find_boundaries(test_word, benden_segmenter.alg1) == ref

    test_word = ['f', 'ʊ', 'r', 'ç', 'ə']
    ref = ['f', 'ʊ', 'r', '+', 'ç', 'ə']
    assert benden_segmenter.find_boundaries(test_word, benden_segmenter.alg1) == ref

    assert benden_segmenter.find_boundaries([], benden_segmenter.alg1) == []


def test_alg2(benden_segmenter):
    test_word = ['yː', 'b', 'ə', 'r', 'm', 'ɔ', 'r', 'g', 'ə', 'n']
    ref = ['yː', 'b', 'ə', 'r', '+', 'm', 'ɔ', 'r', 'g', 'ə', 'n']
    assert benden_segmenter.find_boundaries(test_word, benden_segmenter.alg2) == ref

    test_word = ['f', 'ʊ', 'r', 'ç', 'ə']
    ref = ['f', 'ʊ', 'r', '+', 'ç', 'ə']
    assert benden_segmenter.find_boundaries(test_word, benden_segmenter.alg2) == ref

    assert benden_segmenter.find_boundaries([], benden_segmenter.alg2) == []


def test_alg3(benden_segmenter):
    test_word = ['yː', 'b', 'ə', 'r', 'm', 'ɔ', 'r', 'g', 'ə', 'n']
    ref = ['yː', 'b', 'ə', 'r', '+', 'm', 'ɔ', 'r', 'g', 'ə', 'n']
    assert benden_segmenter.find_boundaries(test_word, benden_segmenter.alg3) == ref

    test_word = ['f', 'ʊ', 'r', 'ç', 'ə']
    ref = ['f', 'ʊ', 'r', 'ç', '+', 'ə']
    assert benden_segmenter.find_boundaries(test_word, benden_segmenter.alg3) == ref

    assert benden_segmenter.find_boundaries([], benden_segmenter.alg3) == []


def test_str_input(benden_segmenter):
    # method should be able to internally convert a string to a list
    test_word = ['f', 'ʊ', 'r', 'ç', 'ə']
    test_word_as_str = 'fʊrçə'

    assert (benden_segmenter.find_boundaries(test_word, benden_segmenter.alg3)
            == benden_segmenter.find_boundaries(test_word_as_str, benden_segmenter.alg3))


def test_trie_retrieval(words):
    # split the input words in half and pass them as sequence clusters
    split_idx = int(len(words) / 2)
    sequence_clusters = [words[:split_idx], words[split_idx:]]
    segmenter = BendenMorphemeSegmenter(sequence_clusters)

    # local tries and global trie should be set up and have different content
    assert len(segmenter.tries) == 2
    assert segmenter.global_trie != segmenter.tries[0] != segmenter.tries[1]

    # try to retrieve a trie for a word from the first cluster...
    test_word_1 = words[0]
    trie1 = segmenter.get_trie_for_word(test_word_1)
    assert trie1 == segmenter.tries[0]

    # ...and the second cluster
    test_word_2 = words[-1]
    trie2 = segmenter.get_trie_for_word(test_word_2)
    assert trie2 == segmenter.tries[1]
    assert trie2 != trie1

    # try to retrieve a word that is not in the lexicon; should return the global trie
    test_word = ['x', 'y', 'z']
    trie = segmenter.get_trie_for_word(test_word)
    assert trie == segmenter.global_trie


def test_format_string(benden_segmenter):
    test_word = ['f', 'ʊ', 'r', 'ç', '+', 'ə']
    test_word2 = ['f', 'ʊ', 'r', 'ç', '+', 'ə', '+']
    test_word3 = ['f', 'ʊ', 'r', 'ç', '+', 'ə', '+', '+']
    test_word4 = ['+', '+', 'f', 'ʊ', 'r', 'ç', '+', 'ə', '+', '+']
    test_word_as_str = 'f ʊ r ç + ə'
    assert (benden_segmenter.format_result_string(test_word)
            == benden_segmenter.format_result_string(test_word2)
            == benden_segmenter.format_result_string(test_word3)
            == benden_segmenter.format_result_string(test_word4)
            == test_word_as_str)


