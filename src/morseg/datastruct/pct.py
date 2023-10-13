from morseg.datastruct.trie import Trie, TrieNode
from morseg.utils.list_utils import reverse_list


class PCT(Trie):
    """
    An implementation of a Patricia Compact Trie (PCT) that can learn and store probability distributions
    of affixes in its nodes, as described in Bordag (2008).
    """
    def __init__(self, reverse=False):
        super().__init__()
        self.__initialize_root()
        self.reverse = reverse

    def __initialize_root(self):
        self.root = PCTNode("")

    def insert(self, word, affix=None):
        """Insert a word into the trie"""
        if self.reverse:
            word = reverse_list(word)

        word = self.sanitize_input(word)

        # loop through each character in the word and add/update the node respectively
        node = self.root

        for char in word:
            node = node.add_child(char, affix)

    def insert_all(self, words):
        for word, affix in words:
            self.insert(word, affix)

    def __get_deepest_matching_node(self, word):
        node = self.root

        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                break

        return node

    def get_affix_probabilities(self, word):
        if self.reverse:
            word = reverse_list(word)
        
        deepest_matching_node = self.__get_deepest_matching_node(word)

        return deepest_matching_node.get_probability_distribution()
    
    def query(self, x):
        if self.reverse:
            x = reverse_list(x)
            
        query_result = super().query(x)

        if self.reverse:
            query_result = [(reverse_list(word), count) for word, count in query_result]

        return query_result
    
    def get_successor_values(self, word):
        if self.reverse:
            word = reverse_list(word)
        
        result = super().get_successor_values(word)

        if self.reverse:
            result = reverse_list(result)

        return result

    def __eq__(self, other):
        return super().__eq__(other) and self.reverse == other.reverse


class PCTNode(TrieNode):
    def __init__(self, char):
        super().__init__(char)

        # important: empty string is a valid key for affix counts, indicating that there is no affix!
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
