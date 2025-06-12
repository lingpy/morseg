from morseg.utils.wrappers import WordWrapper, WordlistWrapper
from linse.typedsequence import Word, Morpheme

import pytest


@pytest.fixture
def w():
    w = [["t", "e"], ["s", "t"]]
    return WordWrapper(w)


def test_word_wrapper_init(w):
    assert isinstance(w, Word)
    assert len(w) == 1
    assert len(w[0]) == 4
    assert len(w.gold_segmented) == 2
    assert len(w.unsegmented) == 1
    assert len(w.unsegmented[0]) == 4


def test_word_wrapper_copy(w):
    w2 = w.copy()
    assert w == w2
    assert w.unsegmented == w2.unsegmented
    assert w.gold_segmented == w2.gold_segmented

    # modify copy
    w2.split(1)
    assert w != w2
    assert w.unsegmented == w2.unsegmented
    assert w.gold_segmented == w2.gold_segmented

    # update
    w.update(w2)
    assert w == w2
    assert len(w) == 2
    assert w.unsegmented == w2.unsegmented
    assert w.gold_segmented == w2.gold_segmented


def test_word_wrapper_splits(w):
    # nothing should happen when splits are out of bounds
    w.split(0)
    assert len(w) == 1
    w.split(4)
    assert len(w) == 1
    assert w.get_splits() == []

    # insert an actual split
    w.split(1)
    assert len(w) == 2
    assert w.has_split_at(1)
    assert not w.has_split_at(2)
    w.split(2)
    assert len(w) == 3
    assert w.has_split_at(2)

    # get splits
    assert w.get_splits() == [1, 2]
    assert w.get_splits(ignore_token="t") == [1]

    # get gold split
    assert w.get_gold_splits() == [2]


def test_word_wrapper_merges(w):
    w.split_everywhere()
    assert len(w) == 4

    w.merge(Morpheme("t"), Morpheme("e"))
    assert len(w) == 3

    w.remove_split(2)
    assert len(w) == 2
    assert w.get_splits() == [3]

    # nothing should happen here
    w.remove_split(0)
    w.remove_split(1)
    w.remove_split(4)
    assert len(w) == 2
    assert w.get_splits() == [3]


def test_word_wrapper_merges_with_wp_token(w):
    w.split_everywhere()
    w.add_wp_token(wp_token="##")
    assert len(w) == 4

    w.merge(Morpheme("t"), Morpheme(["##", "e"]), wp_token="##")
    assert len(w) == 3
    assert w[0] == ["t", "e"]

    assert w[1] == ["##", "s"]
    w.remove_wp_token(wp_token="##")
    assert w[1] == ["s"]


def test_hash(w):
    assert hash(w) == hash("[['t', 'e', 's', 't']]")


@pytest.fixture
def wl(test_data):
    return WordlistWrapper.from_file(test_data / "german.tsv")


def test_read_wl(wl):
    assert len(wl) == 40
    # actual objects should have no segmentations (only the Gold standard)
    assert all(len(x) == 1 for x in wl)
    assert len(wl[12].gold_segmented) == 2


def test_wl(wl):
    assert wl[Word([["f", "ʏ", "n", "f"]])]

    # get all unsegmented forms
    unsegmented = list(wl.unsegmented())
    assert len(unsegmented) == 40
    assert all(len(x) == 1 for x in unsegmented)

    # get all gold segmented forms
    gold_segmented = list(wl.gold_segmented())
    assert len(gold_segmented) == 40
    assert not all(len(x) == 1 for x in gold_segmented)


def test_wl_split_everywhere(wl):
    wl.split_everywhere()

    for word in wl:
        for morpheme in word:
            assert len(morpheme) == 1


def test_wl_copy(wl, w):
    wl2 = wl.copy()
    assert wl == wl2
    wl2[0] = w
    assert wl != wl2


def test_wl_merge_and_split(wl):
    wl.split_everywhere()
    wl.merge(Morpheme(["ts"]), Morpheme(["v"]))
    assert len(wl[1][0]) == 2
    assert len(wl[11][0]) == 2


def test_wl_wp_token(wl):
    wl.split_everywhere()
    wl.add_wp_token(wp_token="##")
    wl.merge(Morpheme(["ts"]), Morpheme(["##", "v"]), wp_token="##")
    assert len(wl[1][0]) == 2
    assert len(wl[11][0]) == 2

    # every non-initial subword should start with that special token
    for word in wl:
        for morpheme in word[1:]:
            assert morpheme[0] == "##"

    # remove special token, it should not be found anywhere anymore
    wl.remove_wp_token(wp_token="##")
    for word in wl:
        for morpheme in word:
            for segment in morpheme:
                assert segment != "##"


def test_wl_unigram_counts(wl):
    wl.split_everywhere()
    counts = wl.unigram_counts()
    assert counts[Morpheme("œ")] == 1
    assert counts[Morpheme("ʏ")] == 4


def test_wl_bigram_counts(wl):
    wl.split_everywhere()
    counts = wl.bigram_counts()
    assert counts[(Morpheme("f"), Morpheme("ʏ"))] == 4


def test_wl_f1score(wl):
    # completely unsegmented: precision and recall should be 0
    assert wl.f1_score() == (0, 0, 0)

    # add 1 correct split: precision should be 1, recall very low
    wl[12].split(3)
    f1, prec, recall = wl.f1_score()
    assert prec == 1
    assert 0 < recall < 0.1
    assert 0 < f1 < 0.1

    # split everywhere; recall should be 1, precision fairly low
    wl.split_everywhere()
    f1, prec, recall = wl.f1_score()
    assert recall == 1
    assert 0 < prec < 0.5
    assert 0 < f1 < 0.5


