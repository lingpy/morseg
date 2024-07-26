"""
Tokenizers are methods that work with pure wordlists.
"""
from typing import List
import random
from linse.typedsequence import Word
from morseg.utils.wrappers import WordWrapper, WordlistWrapper

import collections

try:
    import morfessor
except ImportError:
    morfessor = False


class Tokenizer:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _preprocess(self, words: WordlistWrapper):
        self.training_data = self.forms = words.copy()

    def _train(self, **kwargs):
        pass

    def _postprocess(self):
        pass

    def train(
            self, 
            words: WordlistWrapper,
            **kwargs):
        self._preprocess(words)
        self._train(**kwargs)
        self._postprocess()
    
    def _tokenize(self, word, **kwargs):
        return self.forms[word]

    def __call__(self, word: Word, **kwargs) -> Word:
        return self._tokenize(word, **kwargs)

    def tokenize(
            self, 
            words: List[Word], 
            **kwargs
            ):
        for word in words:
            yield self(word, **kwargs)

    def get_segmentations(self):
        for form in self.forms:
            yield form


class RandomTokenizer(Tokenizer):
    """
    Tokenize words randomly.

    Notes
    -----
    This tokenizer proceeds in a very simple fashion by using one parameter
    that decides about the splitting of a word into morphemes. This parameter
    decides about the maximum number of break points in relation to a word.
    """

    def __init__(self, morpheme_ratio=0.5):
        Tokenizer.__init__(self, morpheme_ratio=morpheme_ratio)

    def _tokenize(self, word: Word, **kwargs):
        # get number of break points
        new_word = []
        for morpheme in word:
            new_word += list(morpheme)
        idxs = list(range(len(new_word)))
        break_point_number = random.randint(
                0, 
                int((len(new_word) - 2) * self.kwargs["morpheme_ratio"] + 0.5))
        break_points = random.sample(idxs[1:-1], break_point_number)
        out = Word([""])
        for i in range(len(new_word)):
            if i in break_points:
                out.append(new_word[i])
            else:
                out[-1].append(new_word[i])
        return out


class PairEncoding(Tokenizer):
    """

    Notes
    -----
    Code taken with modifications from https://www.geeksforgeeks.org/byte-pair-encoding-bpe-in-nlp/
    """
    def __init__(self):
        Tokenizer.__init__(self)

    def _preprocess(self, words):
        self.training_data = self.forms = words
        self.training_data.split_everywhere()
    
    def _train(
            self,
            iterations=60,
            threshold=3
            ):
        # merge most frequent bigram
        for _ in range(iterations):
            pairs = self.training_data.bigram_counts()
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < threshold:
                break
            self.training_data.merge(*best_pair)


class WordPiece(Tokenizer):
    def _preprocess(self, words: WordlistWrapper, wp_prefix="##"):
        self.training_data = self.forms = words
        self.training_data.split_everywhere()
        self.training_data.add_wp_token(wp_token=wp_prefix)

    def _train(self, iterations=60, threshold=0, wp_prefix="##"):
        alphabet = self.training_data.unigram_counts()

        for _ in range(iterations):
            # count bigram frequencies
            bigram_freq = self.training_data.bigram_counts()

            # get pair with best score
            best_score = 0.0
            best_pair = None
            best_pair_freq = 0

            for pair, freq in bigram_freq.items():
                s1, s2 = pair
                score = freq / (alphabet[s1] * alphabet[s2])
                if score > best_score:
                    best_score = score
                    best_pair = pair
                    best_pair_freq = freq

            # stop merging if no score exceeds the threshold, or if there is nothing left to merge
            if best_score < threshold or not best_pair:
                break

            # update alphabet frequencies
            best_first, best_second = best_pair
            alphabet[best_first] -= best_pair_freq
            alphabet[best_second] -= best_pair_freq

            # remove special prefix from second part, add merged pair to the alphabet
            stripped_second = best_second.copy()
            stripped_second.remove(wp_prefix)
            alphabet[best_first + stripped_second] = best_pair_freq

            self.training_data.merge(best_first, best_second, wp_token=wp_prefix)

        # remove special prefix token from vocabulary
        self.training_data.remove_wp_token(wp_token=wp_prefix)


class LetterSuccessorVariety(Tokenizer):

    def _preprocess(self, words: WordlistWrapper):
        self.training_data = self.forms = words

    def _train(self):
        self.sv = collections.defaultdict(lambda : collections.defaultdict(int))
        for word in self.training_data:
            for morpheme in word:
                for sound_a, sound_b in zip(["^"] + morpheme, morpheme + ["$"]):
                    self.sv[sound_a][sound_b] += 1

    def profile(self, word, threshold=1):
        """
        Makes a profile and determines break points.

        The profile is used to break words.
        """
        out = []
        for morpheme in word:
            output = []
            profile = []
            for sound_a, sound_b in zip(morpheme[:-1], morpheme[1:]):
                profile += [self.sv[sound_a][sound_b]]
            # determine peaks (i < j) as our break points
            break_points = [0]
            for idx, (i, j) in enumerate(zip(profile[:-1], profile[1:])):
                if j - i >= threshold:
                    break_points += [idx + 1]
            break_points += [len(morpheme)]
            out += [(profile, break_points)]
        return out

    def __call__(self, word: Word, threshold=1):
        break_points = self.profile(word, threshold=threshold)
        out = []
        for morpheme, (_, bp) in zip(word, break_points):
            for (i, j) in zip(bp[:-1], bp[1:]):
                out += [morpheme[i:j]]
        return Word(out)



class Morfessor(Tokenizer):
    def _preprocess(self, words: WordlistWrapper):
        if not morfessor:
            raise ValueError("You must install the morfessor software package")
        self.forms = words.copy()
        self.training_data = [(1, tuple(m[0])) for m in words.unsegmented()]

    def _train(self, **kwargs):
        self.model = morfessor.BaselineModel()
        self.model.load_data(self.training_data)
        self.model.train_batch(**kwargs)

    def _postprocess(self):
        """
        Store segmentations from Morfessor in own model.
        """
        for f in self.forms:
            res = self.model.segment(tuple(f.unsegmented[0]))
            f.update(res)
