import math

from lingpy.align import Multiple
from itertools import combinations
from collections import Counter
from segmenter import Segmenter
from morseg.utils import *

# TODO the actually harder part here is the pre-selection of the sequence sets
# In the original study, this is done by clustering together words by semantic and phonetic similarity


class KirschenbaumMorphemeSegmenter(Segmenter):
    """
    segments morphemes for given set of sequences, as described in Kirschenbaum (2013).
    """
    def __init__(self, sequences: list, scoring_scheme=None):
        """
        initialize segmenter.
        :param sequences: the sequences to be segmented, represented of threefold-nested list (segments -> words -> clusters)
        :param scoring_scheme: which of the two pattern scoring schemes to use, defaults to Method A
        """
        super().__init__(sequences)
        self.scores = {}  # inferred scores for patterns will be stored here (pattern -> list of scores)
        # note: since patterns are represented as lists, which are not hashable, lookup keys are generated
        # by joining the respective list to blank spaces: " ".join(pattern)

        for sequence_cluster in sequences:
            scorer = Scorer(sequence_cluster)
            if scoring_scheme == "B" or scoring_scheme == "B":
                pattern, score = scorer.method_b()
            else:
                pattern, score = scorer.method_a()
            pattern_key = " ".join(pattern)

            if pattern_key in self.scores:
                self.scores[pattern_key].append(score)
            else:
                self.scores[pattern_key] = [score]

    def match(self, word):
        """
        matches all patterns that apply to a given word
        :param word: the queried word
        :return: all applicable patterns
        """
        applicable_patterns = []
        for pattern in self.scores.keys():
            if list_contains_sublist(word, pattern):
                applicable_patterns.append(pattern.split(" "))

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
            sorted_patterns = sorted(applicable_patterns, key=lambda x: sum(self.scores[get_key_for_pattern(x)]), reverse=True)
        else:
            sorted_patterns = sorted(applicable_patterns, key=lambda x: len(self.scores[get_key_for_pattern(x)]), reverse=True)

        stem = word
        applied_patterns = []

        # subsequently apply best ranking patterns and replace them in order to
        # avoid application of conflicting patterns.
        # the counter of applied patterns serves as placeholder to enable correct resubstitution later.
        num_applied_patterns = 0
        for pattern in sorted_patterns:
            while list_contains_sublist(stem, pattern):
                stem = remove_and_insert_placeholder(stem, pattern, num_applied_patterns)

        # re-insert applied patterns (i.e. detected morphs) with dashes to indicate morpheme boundaries
        segmented_word = stem
        for i, pattern in enumerate(applied_patterns):
            # add boundary symbols at the beginning and the end of the pattern that has been identified as morpheme
            pattern.append(self.BOUNDARY_SYMBOL)
            pattern.insert(0, self.BOUNDARY_SYMBOL)
            segmented_word = insert_and_remove_placeholder(segmented_word, pattern, i)

        # clean up leading, trailing and double boundary symbols from string
        segmented_word = remove_repeating_symbols(segmented_word, self.BOUNDARY_SYMBOL)
        if segmented_word[0] == self.BOUNDARY_SYMBOL:
            segmented_word = segmented_word[1:]
        if segmented_word[-1] == self.BOUNDARY_SYMBOL:
            segmented_word = segmented_word[:-1]

        return segmented_word


def get_key_for_pattern(pattern):
    return " ".join(pattern)


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
    # sequences = ['woldemort','waldemar','wladimir','vladymyr']
    # segmenter = Scorer(sequences)

    list_a = ["a", "b", "c", "d"]
    list_b = ["b", "c", "d", "e", "f"]

    print(list_contains_sublist(list_a, list_b))
