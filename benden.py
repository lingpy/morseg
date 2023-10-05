import math
from trie import Trie
from cldf_utils import get_segments_by_language

BOUNDARY_SYMBOL = "-"


class BendenMorphemeSegmentator(object):
    """
    A class providing algorithms for morpheme segmentation as described in Benden (2005)
    """

    # TODO ensure correct identification of local maxima
    # (cases where there is a non-maximum plataeu cause trouble)
    def __init__(self, forms):
        # TODO get segments from database
        # potentially integrate separate trie models per language in database
        self.trie = Trie()
        self.trie.insert_all(forms)

    def alg1(self, word):
        boundary_indices = []
        svs = self.trie.get_successor_values(word)

        print(svs)

        prev_sv = -1
        for i, (_, sv) in enumerate(svs):
            # set a morpheme boundary after every segment that has a higher SV than its predecessor
            if sv > prev_sv != -1:
                boundary_indices.append(i)
            prev_sv = sv

        return boundary_indices

    def alg2(self, word):
        boundary_indices = []
        svs = self.trie.get_successor_values(word)

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

    def alg3(self, word):
        boundary_indices = []
        svs = self.trie.get_successor_values(word)

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


def find_boundaries(word, algorithm):
    boundary_indices = algorithm(word)

    # make sure word is represented as list
    if isinstance(word, str):
        word = [*word]

    # copy list in order not to modify the original object
    word = word.copy()

    for n, i in enumerate(boundary_indices):
        # n is the number of boundaries that has already been added to the word, has to be added to the original target index
        word.insert(n+i+1, BOUNDARY_SYMBOL)

    return word


def format_result_string(result):
    return "".join(result).strip("-")


if __name__ == "__main__":
    segments_by_language = get_segments_by_language()
    test_segments = segments_by_language["german"]

    segmentator = BendenMorphemeSegmentator(test_segments)

    test_word = ["ɡ", "ʁ", "oː", "s", "f", "aː", "t", "ɐ"]
    # ʃpiːlən

    for word in test_segments:
        print(format_result_string(find_boundaries(word, segmentator.alg1)))
        print(format_result_string(find_boundaries(word, segmentator.alg2)))
        print(format_result_string(find_boundaries(word, segmentator.alg3)))
        print("\n\n")


