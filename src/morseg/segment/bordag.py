from morseg.segment.segmenter import Segmenter
from morseg.segment.benden import BendenMorphemeSegmenter
from morseg.datastruct.pct import PCT


"""
Workflow:

VANILLA
1. preliminary segmentation by LSV (ideally from words with similar function, maybe POS tags?)
2. use these segmentations to "train" prefix and suffix tries

REFINED
1. split compounds
2. iterate LSV algorithm
3. refine trie classification by the means of certain restrictions

"""


class BordagSegmenter(Segmenter):
    def __init__(self, sequences, **kwargs):
        super().__init__(sequences, **kwargs)
        self.prefix_pct = PCT()  # called 'forward trie' in the paper
        self.suffix_pct = PCT(reverse=True)  # called 'backward trie' in the paper
        # Benden Segmenter essentially implements LSV segmentation in 3 different flavours
        self.lsv_segmenter = BendenMorphemeSegmenter(sequences)

    def insert_to_trie(self, word: list):
        if not (word and self.BOUNDARY_SYMBOL in word):
            return

        # 'peel off' affixes recursively, ordered by the following hierarchy:
        # 1. outermost affixes take preference
        # 2. shorter affixes take preference
        # 3. suffixes take preference over prefixes

        first_boundary_index = word.index(self.BOUNDARY_SYMBOL)
        last_boundary_index = len(word) - 1 - word[::-1].index(self.BOUNDARY_SYMBOL)

        # split off potential prefix and suffix
        prefix = word[:first_boundary_index]
        suffix = word[last_boundary_index+1:]

        # define the affix to be split off and the remaining stem according to the hierarchy above,
        # insert word and affix to the corresponding trie (insert method of PCT ignores boundary symbols,
        # so no need to remove them manually here)
        if len(prefix) < len(suffix):
            remaining_stem = word[first_boundary_index+1:]
            self.prefix_pct.insert(word, prefix)
        else:
            remaining_stem = word[last_boundary_index:]
            self.suffix_pct.insert(word, suffix)

        # recursive call to the method to process the remaining stem
        self.insert_to_trie(remaining_stem)

    def train_segmenter_vanilla(self):
        # step 1: get lsv segmentations as first input to train tries
        lsv_segmented_words = self.lsv_segmenter.segment_all()

        # step 2: train prefix and suffix tries by inserting the words
        for cluster in lsv_segmented_words:
            for word in cluster:
                self.insert_to_trie(word)

    def train_segmenter_refined(self):
        """
        1. split compounds
            PROBLEM: candidate selection is not properly described in paper (maybe exhaustive search on a small sample
                        according to the described scoring function)
            PROBLEM: frequency heuristics are most probably not applicable to wordlists
        2. iterate LSV algorithm
            PROBLEM: paper speaks of a LSV threshold which is never defined
        3. refine trie classification by the means of certain restrictions
            PROBLEM: heuristics probably not directly applicable
        """
        pass
