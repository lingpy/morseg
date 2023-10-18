import math
from morseg.datastruct.trie import Trie
from .segmenter import *


class BendenMorphemeSegmenter(Segmenter):
    """
    A class providing algorithms for morpheme segmentation as described in Benden (2005)
    """

    def __init__(self, sequences):
        super().__init__(sequences)
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

        print(svs)

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


def format_result_string(result):
    return "".join(result).strip("-")


if __name__ == "__main__":
    # TODO replace by tests and get rid of direct pycldf usage
    # (use German.tsv test file instead and open it directly with LingPy)

    """
    segments_by_language = get_segments_by_language()
    test_segments = [segments_by_language["german"]]

    segmenter = BendenMorphemeSegmenter(test_segments)

    # test_word = ["ɡ", "ʁ", "oː", "s", "f", "aː", "t", "ɐ"]
    # ʃpiːlən

    for cluster in test_segments:
        for word in cluster:
            print(format_result_string(segmenter.find_boundaries(word, segmenter.alg1)))
            print(format_result_string(segmenter.find_boundaries(word, segmenter.alg2)))
            print(format_result_string(segmenter.find_boundaries(word, segmenter.alg3)))
            print("\n\n")
    """


