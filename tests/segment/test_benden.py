import pytest
from morseg.segment.benden import BendenMorphemeSegmenter
from lingpy import Wordlist


@pytest.fixture
def words(test_data):
    wl = Wordlist(str(test_data / "German.tsv"))
    return [wl[idx, "tokens"] for idx in wl]


def test_init(words):
    segmenter = BendenMorphemeSegmenter([words])
    assert len(segmenter.tries) == 1
    assert segmenter.tries[0] == segmenter.global_trie
