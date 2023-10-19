class Segmenter(object):
    def __init__(self, sequences, **kwargs):
        """
        Initialize the segmenter.
        :param sequences: the sequences to be segmented, represented as a threefold-nested list of strings.
                    the lists represent the following information, from innermost to outermost:
                    segments (sounds/letters) -> words -> word clusters
        """
        self.sequences = sequences
        self.BOUNDARY_SYMBOL = kwargs["boundary_symbol"] if "boundary_symbol" in kwargs else "+"
        self.EOS_SYMBOL = kwargs["eos_symbol"] if "eos_symbol" in kwargs else "#"  # end-of-sequence symbol

    def segment_all(self):
        """
        a method stub for segmenting all given sequences, to be overriden in the subclasses.
        :return: the segmented sequences.
        """
        return self.sequences
