import math

from lingpy.align import Multiple
from itertools import combinations
from collections import Counter

# TODO the actually harder part here is the pre-selection of the sequence sets
# In the original study, this is done by clustering together words by semantic and phonetic similarity


class KirschenbaumMorphemeSegmentator(object):
    """
    segments morphemes for given set of sequences, as described in Kirschenbaum (2013).
    """
    def __init__(self, database):
        # TODO: should be able to read relevant information from CLDF database and cluster sequences for alignment
        self.clusters = []
        self.scores = {}  # inferred scores for patterns will be stored here (pattern -> list of scores)

    def match(self, word):
        """
        matches all patterns that apply to a given word
        :param word: the queried word
        :return: all applicable patterns
        """
        applicable_patterns = []
        for pattern in self.scores.keys():
            if pattern in word:
                applicable_patterns.append(pattern)

        return applicable_patterns

    def get_best_segmentation(self, word, score_based=True):
        """
        retrieve the best possible segmentation for the queried word.
        :param word: the queried word
        :param score_based: if True, rank potential morphs by local scores; otherwise, by raw frequency.
        :return: the segmented word
        """
        applicable_patterns = self.match(word)
        # sort applicable patterns by ranking scheme
        if score_based:
            sorted_patterns = sorted(applicable_patterns, key=lambda x: sum(self.scores[x]), reverse=True)
        else:
            sorted_patterns = sorted(applicable_patterns, key=lambda x: len(self.scores[x]), reverse=True)

        stem = word
        applied_patterns = []

        # subsequently apply best ranking patterns and replace them in order to
        # avoid application of conflicting patterns
        num_applied_patterns = 0
        for pattern in sorted_patterns:
            while pattern in stem:
                stem = stem.replace(pattern, f"<#{num_applied_patterns}>", 1)

        # re-insert applied patterns (i.e. detected morphs) with dashes to indicate morpheme boundaries
        segmented_word = stem
        for i, pattern in enumerate(applied_patterns):
            segmented_word = segmented_word.replace(f"<#{i}>", f"-{pattern}-")

        # clean up leading, trailing and double dashes from string
        segmented_word = segmented_word.strip("-").replace("--", "-")

        return segmented_word


class Scorer(object):
    """
    subclass for finding and scoring patterns, each Scorer instance corresponds to one MSA
    """
    def __init__(self, seqs):
        self.size = len(seqs)
        self.msa = Multiple(seqs)
        self.msa.prog_align()
        self.patterns = self.find_patterns()
        self.pattern_counter = Counter(self.patterns)

    def find_patterns(self):
        patterns = []

        for al1, al2 in combinations(self.msa.alm_matrix, 2):
            current_pattern = ""
            for s1, s2 in zip(al1, al2):
                if s1 == "-" or s2 == "-" or s1 != s2:
                    if len(current_pattern) > 0:
                        patterns.append(current_pattern)
                    current_pattern = ""
                else:
                    current_pattern += s1

        return patterns

    def method_a(self):
        best_pattern = ""
        best_score = 0.0

        for pattern in set(self.patterns):
            score = 2 / (1 / self.pattern_counter[pattern] + 1 / len(pattern))
            if score > best_score:
                best_pattern = pattern
                best_score = score

        return best_pattern, best_score

    def method_b(self):
        best_pattern = ""
        best_score = 0.0

        total_count = self.pattern_counter.total()

        for pattern in set(self.patterns):
            score = (self.pattern_counter[pattern] / total_count) * math.log(self.size)
            if score > best_score:
                best_pattern = pattern
                best_score = score

        return best_pattern, best_score


if __name__ == "__main__":
    sequences = ['woldemort','waldemar','wladimir','vladymyr']
    segmentator = Scorer(sequences)
