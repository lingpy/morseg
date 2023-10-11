from morseg.datastruct import Trie, TrieNode


def test_init():
    trie = Trie()
    assert isinstance(trie.root, TrieNode)
    assert trie.root.char == ""
