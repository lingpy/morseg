import pytest

from morseg.algorithms.tokenizer import *
from morseg.utils.wrappers import WordlistWrapper


@pytest.fixture
def wl(test_data):
    return WordlistWrapper.from_file(test_data / "german.tsv")


def test_pair_encoding(wl):
    model = PairEncoding()
    model.train(wl, threshold=3, vocab_size=20, iterations=200, callbacks=["alphabet_size", "f1"])
    assert len(model.training_history["alphabet_size"]) == len(model.training_history["f1"])
    assert model.forms.f1_score()[0] == pytest.approx(0.4950, abs=0.001)


def test_wordpiece(wl):
    model = WordPiece()
    model.train(wl, threshold=0.05, vocab_size=20, iterations=200, callbacks=["alphabet_size", "f1"])
    assert len(model.training_history["alphabet_size"]) == len(model.training_history["f1"])
    assert model.forms.f1_score()[0] == pytest.approx(0.4655, abs=0.001)


def test_unigram(wl):
    model = UnigramSentencePiece()
    model.train(wl, vocab_size=20, count_single_characters=False)
    assert model.forms.f1_score()[0] == pytest.approx(0.3596, abs=0.001)


def test_morfessor(wl):
    model = Morfessor()
    model.train(wl)
    assert model.forms.f1_score()[0] == pytest.approx(0.8073, abs=0.001)


def test_affix_tokenization(wl):
    model = LSPVTokenizer(strategy="subword")
    model.train(wl)
    assert model.forms.f1_score()[0] == pytest.approx(0.8037, abs=0.001)


def test_lspv(wl):
    model = LSPVTokenizer(method="type", strategy="peak")
    model.train(wl)
    assert model.forms.f1_score()[0] == pytest.approx(0.6918, abs=0.001)


def test_lsv(wl):
    model = LSVTokenizer(method="type", strategy="peak")
    model.train(wl)
    assert model.forms.f1_score()[0] == pytest.approx(0.6721, abs=0.001)
    model = LSVTokenizer(method="type", strategy="rise")
    model.train(wl)
    assert model.forms.f1_score()[0] == pytest.approx(0.6964, abs=0.001)
    model = LSVTokenizer(method="type", strategy="threshold", threshold=2)
    model.train(wl)
    assert model.forms.f1_score()[0] == pytest.approx(0.3373, abs=0.001)


def test_lpv(wl):
    model = LPVTokenizer(method="type", strategy="peak")
    model.train(wl)
    assert model.forms.f1_score()[0] == pytest.approx(0.7660, abs=0.001)


def test_lspe(wl):
    model = LSPVTokenizer(method="entropy", strategy="peak")
    model.train(wl)
    assert model.forms.f1_score()[0] == pytest.approx(0.6522, abs=0.001)


def test_lspm(wl):
    model = LSPVTokenizer(method="max_drop", strategy="peak")
    model.train(wl)
    assert model.forms.f1_score()[0] == pytest.approx(0.6164, abs=0.001)


def test_nlspv(wl):
    model = LSPVTokenizer(method="normalized", strategy="peak")
    model.train(wl)
    assert model.forms.f1_score()[0] == pytest.approx(0.5143, abs=0.001)


def test_sqentr(wl):
    model = SquareEntropyTokenizer()
    model.train(wl)
    assert model.forms.f1_score()[0] == pytest.approx(0.5299, abs=0.001)

