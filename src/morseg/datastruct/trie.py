from random import shuffle

EOS_SYMBOL = "#"  # symbol to be used to indicate the end of a sequence
BOUNDARY_SYMBOL = "+"


class Trie(object):
    """The trie object"""
    # TODO maybe represent Trie as Compact Trie for improved efficiency.

    def __init__(self):
        """
        The trie has at least the root node.
        The root node does not store any character
        """
        self.__initialize_root()

    def __initialize_root(self):
        self.root = TrieNode("")

    def insert(self, word):
        """Insert a word into the trie"""
        if not word:
            return

        word = self.sanitize_input(word)

        # loop through each character in the word and add/update the node respectively
        node = self.root

        for char in word:
            node = node.add_child(char)

    def sanitize_input(self, word, remove_boundaries=True):
        # make sure word is represented as list
        word = [x for x in word]

        # remove boundary symbols, unless specified otherwise
        if remove_boundaries:
            word = list(filter(lambda x: x != BOUNDARY_SYMBOL, word))

        # make sure eos symbol is not used as segment
        if EOS_SYMBOL in word[:-1]:
            print(f"WARNING: Reserved symbol '{EOS_SYMBOL}' (End-Of-Sequence) used as segment, will be ignored...")
            word = list(filter(lambda segment: segment != EOS_SYMBOL, word))

        # append eos symbol to end of sequence
        if word[-1] != EOS_SYMBOL:
            word.append(EOS_SYMBOL)

        return word

    def insert_all(self, words):
        if not words:
            return

        for w in words:
            self.insert(w)

    def collect_nodes_preorder(self, node=None):
        if node is None:
            node = self.root

        node_list = [node]

        sorted_child_keys = list(sorted(node.children.keys()))

        for child_key in sorted_child_keys:
            node_list.extend(self.collect_nodes_preorder(node.children[child_key]))

        return node_list

    def dfs(self, node, prefix):
        """Depth-first traversal of the trie

        Args:
            - node: the node to start with
            - prefix: the current prefix, for tracing a
                word while traversing the trie
        """
        if node.char == EOS_SYMBOL:
            self.output.append((prefix, node.counter))

        for child in node.children.values():
            if node == self.root:
                self.dfs(child, [])
            else:
                self.dfs(child, prefix + [node.char])

    def query(self, x):
        """Given an input (a prefix), retrieve all words stored in
        the trie with that prefix, sort the words by the number of
        times they have been inserted
        """
        # Use a variable within the class to keep all possible outputs
        # As there can be more than one word with such prefix
        self.output = []
        node = self.root

        # Check if the prefix is in the trie
        for char in x:
            if char in node.children:
                node = node.children[char]
            else:
                # cannot find the prefix, return empty list
                return []

        # Traverse the trie to get all candidates
        self.dfs(node, x[:-1])

        # Sort the results in reverse order and return
        return sorted(self.output, key=lambda x: x[1], reverse=True)

    def get_successor_values(self, word):
        node = self.root
        sv_per_segment = []  # populate with pairs of segment and SV

        for segment in word:
            node = node.children.get(segment)
            if not node:
                break  # populate remaining SVs with 0
            sv = len(node.children)
            sv_per_segment.append((segment, sv))

        while len(sv_per_segment) < len(word):
            i = len(sv_per_segment)
            segment = word[i]
            sv_per_segment.append((segment, 0))
            i += 1

        return sv_per_segment

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False

        nodes = self.collect_nodes_preorder()
        other_nodes = other.collect_nodes_preorder()

        if len(nodes) != len(other_nodes):
            return False

        for node, other_node in zip(nodes, other_nodes):
            if node != other_node:
                return False

        return True


class TrieNode:
    """A node in the trie structure"""

    def __init__(self, char):
        # the character stored in this node
        self.char = char

        # a counter indicating by how many entries the node is matched
        self.counter = 0

        # a dictionary of child nodes
        # keys are characters, values are nodes
        self.children = {}

    def add_child(self, char):
        if char in self.children:
            child_node = self.children[char]
        else:
            child_node = type(self)(char)
            self.children[char] = child_node

        self.counter += 1

        if char == EOS_SYMBOL:
            child_node.counter += 1  # update counter for leaf nodes, since they are never traversed

        return child_node

    def __eq__(self, other):
        if not isinstance(other, TrieNode):
            return False

        return (self.char == other.char and self.counter == other.counter
                and self.children.keys() == other.children.keys())
