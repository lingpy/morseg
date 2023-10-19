from morseg.segment.segmenter import Segmenter


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


