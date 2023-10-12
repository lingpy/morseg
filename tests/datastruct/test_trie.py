from morseg.datastruct import Trie, TrieNode, EOS_SYMBOL
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

    return words


def test_init(trie):
    assert isinstance(trie.root, TrieNode)
    assert trie.root.char == ""


def test_insert(trie):
    word = ["t", "e", "s", "t", "#"]
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
        next_char = word[i]
        node = node.children.get(next_char)
        assert node.char == next_char
        assert node.counter == 1

        # there should not be any children at a node that indicates the end of a sequence
        if node.char == EOS_SYMBOL:
            assert len(node.children) == 0
        else:
            assert len(node.children) == 1
        i += 1


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
    assert EOS_SYMBOL in node.children.keys()


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
    returned_forms = [form for form, count in result]
    assert all(x in ref for x in returned_forms)

    # querying of a full word
    word = ["b", "i", "g", "g", "e", "r"]
    result = trie.query(word)
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
    returned_forms = [form for form, count in result]
    assert all(x in ref for x in returned_forms)

    # empty query should return all words contained in trie
    result = trie.query([])
    assert len(result) == 5
    returned_forms = [form for form, count in result]
    print(returned_forms)
    print(words)
    assert all(x in words for x in returned_forms)

    # add a word for the second time, count should be updated
    word = ["b", "o", "g", "u", "s"]
    trie.insert(word)
    result = trie.query(["b", "o"])
    assert len(result) == 1
    assert result[0][0] == word
    assert result[0][1] == 2


def test_get_successor_values(trie, words):
    pass
