import math
from morseg.datastruct.trie import Trie
from morseg.segment.segmenter import *


class BendenMorphemeSegmenter(Segmenter):
    """
    A class providing algorithms for morpheme segmentation as described in Benden (2005)
    """
    def __init__(self, sequences, **kwargs):
        super().__init__(sequences, **kwargs)
        self.tries = []
        self.global_trie = Trie()

        for sequence_cluster in sequences:
            trie = Trie()
            trie.insert_all(sequence_cluster)
            self.tries.append(trie)
            self.global_trie.insert_all(sequence_cluster)

    def alg1(self, word: list, trie: Trie):
        boundary_indices = []
        svs = trie.get_successor_values(word)

        prev_sv = -1
        for i, (_, sv) in enumerate(svs):
            # set a morpheme boundary after every segment that has a higher SV than its predecessor
            if sv > prev_sv != -1:
                boundary_indices.append(i)
            prev_sv = sv

        return boundary_indices

    def alg2(self, word: list, trie: Trie):
        boundary_indices = []
        svs = trie.get_successor_values(word)

        prev_sv = math.inf
        for i, (_, sv) in enumerate(svs):
            next_sv = svs[i+1][1] if i < len(svs)-1 else 0
            # set a morpheme boundary after the first segment of a local maximum
            # (segment where no immediate neighbor has a higher SV)
            if sv > prev_sv and sv >= next_sv:
                # handling of plateaus
                j = 1
                while sv == next_sv:
                    next_sv = svs[i+j][1] if j < len(word) - i else math.inf
                    j += 1
                if sv > next_sv:
                    boundary_indices.append(i)
            prev_sv = sv

        return boundary_indices

    def alg3(self, word: list, trie: Trie):
        boundary_indices = []
        svs = trie.get_successor_values(word)

        """
        prev_sv = -1
        for i, (_, sv) in enumerate(svs):
            next_sv = svs[i + 1][1] if i < len(svs) - 1 else 0
            # set a morpheme boundary after the last segment of a local maximum
            # (segment where no immediate neighbor has a higher SV)
            if sv >= prev_sv != -1 and sv > next_sv:
                boundary_indices.append(i)
            prev_sv = sv
        """
        # iterate backwards through the word to make sure that the boundary is set at the last position of
        # a local maximum (thus the first one the loop arrives at)
        next_sv = math.inf
        for i in range(len(word)-1, 0, -1):
            position, sv = svs[i]
            prev_sv = svs[i-1][1]
            # set a morpheme boundary after the last segment of a local maximum
            # (segment where no immediate neighbor has a higher SV)
            if sv > next_sv and sv >= prev_sv:
                # handling of plateaus
                j = 1
                while sv == prev_sv:
                    prev_sv = svs[i-j][1] if j < i else math.inf
                    j += 1
                if sv > prev_sv:
                    boundary_indices.insert(0, i)
            next_sv = sv

        return boundary_indices

    def get_trie_for_word(self, word):
        for i, sequence_cluster in enumerate(self.sequences):
            if word in sequence_cluster:
                return self.tries[i]

        return self.global_trie

    def find_boundaries(self, word, algorithm):
        trie = self.get_trie_for_word(word)

        boundary_indices = algorithm(word, trie)

        # make sure word is represented as list
        if isinstance(word, str):
            word = [*word]

        # copy list in order not to modify the original object
        word = word.copy()

        for n, i in enumerate(boundary_indices):
            # n is the number of boundaries that has already been added to the word, has to be added to the original target index
            word.insert(n+i+1, self.BOUNDARY_SYMBOL)

        return word

    def format_result_string(self, result):
        res = " ".join(result)

        # strip leading and trailing boundaries + whitespaces
        prefix_cutoff_index = 0
        for i, char in enumerate(res):
            if char != self.BOUNDARY_SYMBOL and char != " ":
                prefix_cutoff_index = i
                break

        suffix_cutoff_index = len(res)
        for i in range(len(res) - 1, -1, -1):
            char = res[i]
            if char != self.BOUNDARY_SYMBOL and char != " ":
                suffix_cutoff_index = i+1
                break

        return res[prefix_cutoff_index:suffix_cutoff_index]
