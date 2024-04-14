import pytest

from morseg.datastruct.pct import PCT, PCTNode
from morseg.datastruct.trie import Trie, TrieNode
from random import shuffle


@pytest.fixture
def prefix_pct():
    return PCT()


@pytest.fixture
def suffix_pct():
    return PCT(reverse=True)


@pytest.fixture
def prefix_words():
    words = [
        ("pretty", []),
        ("prefix", ["p", "r", "e"]),
        ("fix", ""),
        ("infix", ["i", "n"]),
        ("preface", ["p", "r", "e"])
    ]

    return words


@pytest.fixture
def suffix_words():
    words = [
        ("clear", ""),
        ("clearly", ["l", "y"]),
        ("dearly", ["l", "y"]),
        ("early", ""),
        ("machinery", ["r", "y"])
    ]

    return words


@pytest.fixture
def segmented_words(prefix_words):
    return [([c for c in form], affix) for form, affix in prefix_words]


def test_init(prefix_pct):
    assert isinstance(prefix_pct.root, PCTNode)
    assert prefix_pct.root.char == ""
    assert len(prefix_pct.root.affix_counts) == 0


def test_insert_all(prefix_pct, prefix_words, segmented_words, suffix_pct, suffix_words):
    # inserting words as strings or as lists should both work
    prefix_pct.insert_all(segmented_words)
    other_pct = PCT()
    other_pct.insert_all(prefix_words)
    assert prefix_pct == other_pct

    # 3 possible affixes should be stored in the whole structure (and thus in the root node)
    assert len(prefix_pct.root.affix_counts) == 3

    # words should be inserted in inverse order for suffix tries.
    # all test words end in "r" or "y".
    suffix_pct.insert_all(suffix_words)
    assert len(suffix_pct.root.children) == 2
    assert "y" in suffix_pct.root.children
    assert "r" in suffix_pct.root.children

    # again, 3 possible affixes should be stored in the whole structure (and thus in the root node)
    assert len(suffix_pct.root.affix_counts) == 3

    # check whether the "y" node is populated correctly
    node = suffix_pct.root.children.get("y")
    assert len(node.children) == 2
    assert "l" in node.children
    assert "r" in node.children


def test_suffix_query(suffix_pct, suffix_words):
    # test whether the reverting logic for suffix tries works properly for querying
    suffix_pct.insert_all(suffix_words)
    query_result = suffix_pct.query(["l", "y"])
    assert len(query_result) == 3  # suffix should match 'early', 'dearly', and 'clearly'

    returned_words = [x[0] for x in query_result]
    expected_words = [
        ["e", "a", "r", "l", "y"],
        ["d", "e", "a", "r", "l", "y"],
        ["c", "l", "e", "a", "r", "l", "y"]
    ]

    assert all(word in expected_words for word in returned_words)

    # empty query should match all words
    assert len(suffix_pct.query([])) == 5


def test_suffix_svs(suffix_pct, suffix_words):
    # successor values in suffix tries are essentially predecessor values.
    # boundary cases are already tested in the superclass, so this is only to make sure
    # that the reverting logic works as intended.
    suffix_pct.insert_all(suffix_words)
    expected_result = [("e", 3), ("a", 1), ("r", 1), ("l", 1), ("y", 2)]
    assert suffix_pct.get_successor_values(["e", "a", "r", "l", "y"]) == expected_result


def test_pct_equals_trie(prefix_pct):
    # even with identical shared content, a PCT should never be equal to a Trie
    trie = Trie()
    assert trie != prefix_pct


def test_eq_prefix_pct(prefix_pct, prefix_words):
    # empty tries should be equal
    other_pct = PCT()
    assert prefix_pct == other_pct

    # however, both tries must go in the same direction (i.e. be either suffix or prefix tries)
    other_pct.reverse = True
    assert prefix_pct != other_pct

    # now add content to one of the tries
    other_pct.reverse = False
    prefix_pct.insert_all(prefix_words)
    assert prefix_pct != other_pct

    # add the same words to the other PCT; objects should be equal again
    other_pct.insert_all(prefix_words)
    assert prefix_pct == other_pct

    # the order in which words are added should not matter
    other_pct = PCT()
    shuffle(prefix_words)
    other_pct.insert_all(prefix_words)
    assert prefix_pct == other_pct

    # however, adding different affixes should fail the comparison, even if the base trie structure is identical
    other_pct = PCT()
    word, _ = prefix_words.pop()
    prefix_words.append((word, "AAA"))
    assert prefix_pct != other_pct

    # try whether method is stable against comparison with other object types
    assert prefix_pct != ""
    assert prefix_pct != 1
    assert prefix_pct != {}


def test_eq_suffix_pct(suffix_pct, suffix_words):
    # empty tries should be equal
    other_pct = PCT(reverse=True)
    assert suffix_pct == other_pct

    # now add content to one of the tries
    suffix_pct.insert_all(suffix_words)
    assert suffix_pct != other_pct

    # add the same words to the other PCT; objects should be equal again
    other_pct.insert_all(suffix_words)
    assert suffix_pct == other_pct

    # the order in which words are added should not matter
    other_pct = PCT(reverse=True)
    shuffle(suffix_words)
    other_pct.insert_all(suffix_words)
    assert suffix_pct == other_pct

    # however, adding different affixes should fail the comparison, even if the base trie structure is identical
    other_pct = PCT()
    word, _ = suffix_words.pop()
    suffix_words.append((word, "AAA"))
    assert suffix_pct != other_pct


def test_pct_node_equals_trie_node():
    # even with identical shared content, a PCTNode should never be equal to a TrieNode
    pct_node = PCTNode("")
    trie_node = TrieNode("")

    assert pct_node != trie_node


def test_get_affix_probabilities(suffix_pct, suffix_words):
    suffix_pct.insert_all(suffix_words)
    prob_distribution = suffix_pct.get_affix_probabilities(["e", "a", "s", "i", "l", "y"])
    assert len(prob_distribution) == 2
    assert prob_distribution == [(["l", "y"], 2/3), ([], 1/3)]

    prob_distribution = suffix_pct.get_affix_probabilities(["w", "e", "e", "k"])
    assert len(prob_distribution) == 3
    assert (prob_distribution == [(["l", "y"], 0.4), ([], 0.4), (["r", "y"], 0.2)] or
            prob_distribution == [([], 0.4), (["l", "y"], 0.4), (["r", "y"], 0.2)])
