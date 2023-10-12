from morseg.datastruct.trie import Trie, TrieNode


class PCT(Trie):
    """
    An implementation of a Patricia Compact Trie (PCT) that can learn and store probability distributions
    of affixes in its nodes, as described in Bordag (2008).
    """
    def __init__(self):
        super().__init__()
        self.__initialize_root()

    def __initialize_root(self):
        self.root = PCTNode("", None)

    def insert(self, word, affix=None):
        """Insert a word into the trie"""
        word = self.sanitize_input(word)

        # loop through each character in the word and add/update the node respectively
        node = self.root

        for char in word:
            node = node.add_child(char, affix)

    def insert_all(self, words):
        for word, affix in words:
            self.insert(word, affix)

    def get_deepest_matching_node(self, word):
        node = self.root

        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                break

        return node

    def get_affix_probabilities(self, word):
        deepest_matching_node = self.get_deepest_matching_node(word)

        return deepest_matching_node.get_probability_distribution()


def reverse_input(word):
    return [word[i] for i in range(len(word) - 1, -1, -1)]


class PCTNode(TrieNode):
    def __init__(self, char, affix=None):
        super().__init__(char)

        # important distinction between None (virtual root) and empty string (no affix)!
        if affix is not None:
            self.affix_counts = {affix: 1}
        else:
            self.affix_counts = {}

    def add_child(self, char, affix=None):
        child_node = super().add_child(char)

        if affix is not None:
            if affix in self.affix_counts:
                self.affix_counts[affix] += 1
            else:
                self.affix_counts[affix] = 1

        return child_node

    def get_probability_distribution(self):
        return {affix: (count / self.counter) for affix, count in self.affix_counts.items()}

    def __eq__(self, other):
        if not isinstance(other, PCTNode):
            return False

        return super().__eq__(other) and self.affix_counts == other.affix_counts


if __name__ == "__main__":
    t = PCT()

    """
    words = [
        ["b", "l", "a"],
        ["b", "l", "i"],
        ["b", "l", "u", "b"],
        ["b", "l", "ei"],
        ["b", "l"],
        ["b", "l"]
    ]
    """

    words = [
        ("clear", ""),
        ("clearly", "ly"),
        ("dearly", "ly"),
        ("early", ""),
        ("machinery", "ry")
    ]

    words = [(reverse_input(word), affix) for word, affix in words]

    t.insert_all(words)

    print(t.get_affix_probabilities(reverse_input("easily")))
    print(t.get_affix_probabilities(reverse_input("week")))

    # inserting words as lists works, querying only works as strings
    # print(t.query("bl"))

    # print(t.get_successor_values(["b", "l", "ei"]))
