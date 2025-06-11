from morseg.datastruct import Trie, TrieNode
from morseg.utils.wrappers import WordWrapper
from linse.typedsequence import Morpheme, Word
from random import shuffle

import pytest


@pytest.fixture
def trie():
    return Trie()


@pytest.fixture
def words():
    words = [
        ["b", "i", "n", "g", "o"],
        ["b", "i", "n", "g"],
        ["b", "i", "g"],
        ["b", "i", "g", "g", "e", "r"],
        ["b", "o", "g", "u", "s"]
    ]

    return [WordWrapper(w) for w in words]


def test_init(trie):
    assert isinstance(trie.root, TrieNode)
    assert trie.root.char == ""


def test_insert(trie):
    word = WordWrapper(["t", "e", "s", "t", Trie.EOS_SYMBOL])
    trie.insert(word)

    # check if root node is set up correctly
    assert len(trie.root.children) == 1
    assert trie.root.counter == 1
    assert "t" in trie.root.children

    # check first actual node
    node = trie.root.children.get("t")
    assert node is not None
    assert node.char == "t"
    assert node.counter == 1
    assert len(node.children) == 1

    # check remaining nodes
    i = 1
    while node.children:
        print("CURRENT CHAR: " + node.char)
        print("NEXT ")
        next_char = word.unsegmented[0][i]
        node = node.children.get(next_char)
        assert node.char == next_char
        assert node.counter == 1

        # there should not be any children at a node that indicates the end of a sequence
        if node.char == Trie.EOS_SYMBOL:
            assert len(node.children) == 0
        else:
            assert len(node.children) == 1
        i += 1

    # check whether protected symbols are handled correctly
    word2 = WordWrapper(["t", "e", "s", "t"])
    trie2 = Trie()
    trie2.insert(word2)
    word3 = WordWrapper(["t", "e", Trie.EOS_SYMBOL, Trie.EOS_SYMBOL, "s", "t", Trie.EOS_SYMBOL])
    trie3 = Trie()
    trie3.insert(word3)
    assert trie == trie2 == trie3


def test_insert_all(trie, words):
    trie.insert_all(words)

    # check if all words have been inserted and accounted for
    assert trie.root.counter == 5

    # empty input should not break anything and leave the trie unmodified.
    trie_copy = Trie()
    trie_copy.insert_all(words)
    trie_copy.insert_all([])
    trie_copy.insert_all([[]])
    trie_copy.insert_all(None)
    assert trie == trie_copy

    # check parameters of first node
    node = trie.root.children.get("b")
    assert node.char == "b"
    assert node.counter == 5
    assert len(node.children) == 2
    assert "i" in node.children.keys()
    assert "o" in node.children.keys()

    # check node that links to EOS or continuation
    node = node.children.get("i").children.get("g")
    assert node.char == "g"
    assert len(node.children) == 2
    assert "g" in node.children.keys()
    assert Trie.EOS_SYMBOL in node.children.keys()


def test_eq(trie, words):
    other_trie = Trie()

    # empty tries should be equal
    assert trie == other_trie

    # populate one of the tries, should not be equal anymore
    trie.insert_all(words)
    assert trie != other_trie

    # populate other trie with the same words, should be equal again
    other_trie.insert_all(words)
    assert trie == other_trie

    # adding the same words again results in a different trie, since the counts differ
    other_trie.insert_all(words)
    assert trie != other_trie

    # however, the order in which words are added should not matter
    other_trie = Trie()
    shuffle(words)
    other_trie.insert_all(words)
    assert trie == other_trie

    # comparison to other objects should yield False
    assert trie != ["l", "i", "s", "t"]
    assert trie != "string"
    assert trie != 0
    assert trie != TrieNode("")


def test_query(trie, words):
    trie.insert_all(words)

    # regular prefix querying
    ref = [
        ["b", "i", "n", "g", "o"],
        ["b", "i", "n", "g"],
        ["b", "i", "g"],
        ["b", "i", "g", "g", "e", "r"]
    ]
    result = trie.query(["b", "i"])
    assert len(result) == 4
    assert all(x in ref for x in result)

    # querying of a full word
    word = ["b", "i", "g", "g", "e", "r"]
    result = trie.query(word, freq=True)
    assert len(result) == 1
    assert result[0][0] == word
    assert result[0][1] == 1

    # querying of a full word with potential suffix
    ref = [
        ["b", "i", "g"],
        ["b", "i", "g", "g", "e", "r"]
    ]
    result = trie.query(["b", "i", "g"])
    assert len(result) == 2
    assert all(x in ref for x in result)

    # empty query should return all words contained in trie
    result = trie.query([])
    assert len(result) == 5
    print(result)
    print(words)
    assert all(WordWrapper(x) in words for x in result)

    # should return an empty list if prefix is not found in the trie
    assert trie.query(["a"]) == []

    # add a word for the second time, count should be updated
    word = WordWrapper(["b", "o", "g", "u", "s"])
    trie.insert(word)
    result = trie.query(["b", "o"], freq=True)
    assert len(result) == 1
    assert result[0][0] == word.unsegmented[0]
    assert result[0][1] == 2


def test_get_successor_values(trie, words):
    trie.insert_all(words)

    # check if successor values are populated correctly for existing word
    expected_result = [("b", 2), ("i", 2), ("n", 1), ("g", 2), ("o", 1)]
    assert expected_result == trie.get_successor_values(["b", "i", "n", "g", "o"])

    # check if successor values are populated correctly for partly matching word
    expected_result = [("b", 2), ("o", 1), ("n", 0), ("g", 0), ("o", 0)]
    assert expected_result == trie.get_successor_values(["b", "o", "n", "g", "o"])

    # check if successor values are populated correctly for words that do not match at all
    expected_result = [("d", 0), ("o", 0), ("n", 0), ("g", 0), ("o", 0)]
    assert expected_result == trie.get_successor_values(["d", "o", "n", "g", "o"])


def test_node_equals():
    node = TrieNode("")
    other_node = TrieNode("")

    # empty nodes should be equal
    assert node == other_node

    # update content of one node
    node.add_child("a")
    assert node != other_node

    # update content of other node
    other_node.add_child("a")
    assert node == other_node

    # equality check should also account for the counter
    node.add_child("a")
    assert node != other_node

    # check if comparison is stable against other object types
    assert node != ""
    assert node != 2
    assert node != Trie()


def test_custom_eos_symbol():
    t = Trie(eos_symbol="!")
    assert t.EOS_SYMBOL == "!"


def test_init_with_words(words):
    t = Trie(words=words)
    # check if all words have been inserted and accounted for
    assert t.root.counter == 5

    # check parameters of first node
    node = t.root.children.get("b")
    assert node.char == "b"
    assert node.counter == 5
    assert len(node.children) == 2
    assert "i" in node.children.keys()
    assert "o" in node.children.keys()


def test_input_datatype(trie):
    with pytest.raises(TypeError):
        trie.insert(1)

    with pytest.raises(TypeError):
        trie.insert(["l", "i", "s", "t"])

    with pytest.raises(TypeError):
        trie.insert("string")


def test_reverse_trie(words):
    trie = Trie(reverse=True)
    trie.insert_all(words)

    assert trie.query(["o", "g", "n", "i", "b"])
    assert "b" not in trie.root.children


def test_token_variety(trie, words):
    trie.insert_all(words)

    # query forward trie
    token_var = trie.get_token_variety(words[0]) # b i n g o
    assert len(token_var) == len(words[0].unsegmented[0]) + 1
    assert token_var[0] == [5]
    assert set(token_var[1]) == {4, 1}
    assert token_var[-1] == [1]

    # query backward trie
    t_backwards = Trie(reverse=True)
    t_backwards.insert_all(words)
    # after the first letter, there should be no variety (bingo is the only word ending in 'o')
    assert t_backwards.get_token_variety(words[0])[1:] == 5 * [[1]]

    # query forward trie with word that is not occurring
    word = WordWrapper(["b", "o", "n", "u", "s"])
    token_var = trie.get_token_variety(word)
    assert len(token_var) == len(word.unsegmented[0]) + 1
    assert token_var[0] == [5]
    assert set(token_var[1]) == {4, 1}
    assert token_var[2] == [1]
    assert token_var[3:] == 3 * [[0]]


def test_is_branching(trie, words):
    trie.insert_all(words)

    assert trie.is_branching(Morpheme(["b"]))
    assert not trie.is_branching(Morpheme(["b", "o"]))
    # should return False for unknown prefixes
    assert not trie.is_branching(Morpheme(["p"]))


def test_get_count(trie, words):
    trie.insert_all(words)

    assert trie.get_count(Morpheme(["b"])) == 5
    assert trie.get_count(Morpheme(["b", "o"])) == 1
    assert trie.get_count(Morpheme(["b", "i"])) == 4
    assert trie.get_count(Morpheme(["b", "u"])) == 0


def test_get_node(trie, words):
    trie.insert_all(words)

    assert (trie._get_node(Morpheme(["b", "o"])) == trie._get_node(Word([["b", "o"]])))
    assert trie._get_node(Morpheme(["b", "u"])) is None


def test_get_subwords(trie, words):
    trie.insert_all(words)

    w = Word([["b", "i", "g", "g", "e", "r"]])
    wr = WordWrapper(w)

    assert trie.get_subwords(w) == trie.get_subwords(wr)
    assert len(trie.get_subwords(w)) == 2
    assert ["b", "i", "g"] in trie.get_subwords(w)

    assert ["b", "i", "g"] in trie.get_subwords(Word(["b", "i", "g", "p"]))
