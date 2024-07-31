"""
Tokenizers are methods that work with pure wordlists.
"""
import math

from typing import List
import random
from linse.typedsequence import Word
from morseg.utils.wrappers import WordWrapper, WordlistWrapper
from morseg.datastruct import Trie

import collections

try:
    import morfessor
except ImportError:
    morfessor = False


class Tokenizer:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def _copy_forms(self, words: WordlistWrapper):
        self.forms = words.copy()

    def _preprocess(self):
        self.training_data = self.forms

    def _train(self, **kwargs):
        pass

    def _postprocess(self):
        pass

    def train(
            self, 
            words: WordlistWrapper,
            **kwargs):
        self._copy_forms(words)
        self._preprocess()
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

    def _preprocess(self):
        self.training_data = self.forms
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
    def _preprocess(self, wp_prefix="##"):
        self.training_data = self.forms
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

    def _preprocess(self):
        self.training_data = self.forms

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
    def _preprocess(self):
        if not morfessor:
            raise ValueError("You must install the morfessor software package")
        self.training_data = [(1, tuple(m[0])) for m in self.forms.unsegmented()]

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


class LSVTokenizer(Tokenizer):
    # the possible values for each parameter.
    # the first value doubles as default option.
    param_options = {
        "method": ["type", "token", "entropy", "max_drop", "normalized"],
        "strategy": ["peak", "rise", "threshold"],
        "direction": ["forward", "backward", "both"]
    }

    def __init__(self, **kwargs):
        self.params = {}

        for param, values in self.param_options.items():
            if param in kwargs:
                if kwargs[param] in values:
                    self.params[param] = kwargs[param]
                else:
                    raise ValueError(f"Invalid value for argument {param}: '{kwargs[param]}'")
            else:
                self.params[param] = values[0]

        if self.params["strategy"] == "threshold":
            if "threshold" in kwargs:
                self.params["threshold"] = kwargs["threshold"]
            else:
                raise ValueError("A threshold is required for the threshold segmentation strategy.")

        super().__init__(**kwargs)

    def _preprocess(self):
        # TODO support for backward or bidirectional tries
        self.training_data = Trie(self.forms)

    def _calculate_type_variety(self, token_variety: list):
        return [len(x) for x in token_variety]

    def _calculate_token_variety(self, token_variety: list):
        return token_variety

    def _calculate_successor_entropy(self, token_variety: list):
        entropies = []

        for varieties in token_variety:
            entropy = 0.0
            sum_varieties = sum(varieties)
            for v in varieties:
                p = v / sum_varieties
                entropy -= p * math.log2(p)
            entropies.append(entropy)

        return entropies

    def _calculate_successor_max_drop(self, token_variety: list):
        pass

    def _calculate_norm_lsv(self, token_variety: list):
        pass

    def _train(self, **kwargs):
        # cache segmentations with specified parameters
        self.token_varieties = {word.unsegmented: self.training_data.get_token_variety(word) for word in self.forms}

        var_func_mapping = {
            "type": self._calculate_type_variety,
            "token": self._calculate_token_variety,
            "entropy": self._calculate_successor_entropy,
            "max_drop": self._calculate_successor_max_drop,
            "normalized": self._calculate_norm_lsv
        }

        # get the corresponding function to calculate variety values
        var_func = var_func_mapping.get(self.params["method"], self._calculate_type_variety)

        # calculate variety values for each word
        self.varieties = {}
        for word, token_var in self.token_varieties.items():
            self.varieties[word] = var_func(token_var)

    def _get_splits_at_peak(self, varieties):
        """
        Peak and plateau segmentation strategy, as formalized by Hafer and Weiss (1974).
        Splits a word at index i iff the varieties[i] >= varieties[i+1] and varieties[i] >= varieties[i-1].
        Does not introduce splits if LSV == 1 for type frequency.
        """
        splits = []

        var_threshold = 1 if self.params["method"] == "type" else 0

        for i in range(1, len(varieties)-1):
            if varieties[i] > var_threshold and varieties[i] >= varieties[i-1] and varieties[i] >= varieties[i+1]:
                splits.append(i)

        return splits

    def _get_splits_at_rise(self, varieties):
        """
        Benden (2005)'s **First Algorithm**.
        Introduces a split wherever the variety value is higher than its immediate predecessor.
        """
        splits = []

        for i in range(1, len(varieties)):
            if varieties[i] > varieties[i-1]:
                splits.append(i)

        return splits

    def _get_splits_by_threshold(self, varieties):
        splits = []

        for i, variety in enumerate(varieties):
            if variety > self.params["threshold"]:
                splits.append(i)

        return splits

    def _postprocess(self):
        split_func_mapping = {
            "peak": self._get_splits_at_peak,
            "rise": self._get_splits_at_rise,
            "threshold": self._get_splits_by_threshold
        }

        split_func = split_func_mapping.get(self.params["strategy"], self._get_splits_at_peak)

        for word, varieties in self.varieties.items():
            splits = split_func(varieties)
            for i in splits:
                self.forms[word].split(i)

    def _tokenize(self, word, **kwargs):
        if (kwargs.get("method", self.params["method"]) == self.params["method"] and
                kwargs.get("strategy", self.params["strategy"]) == self.params["strategy"]):
            return super()._tokenize(word)  # returns the cached segmentation
        else:
            # TODO calculate segmentation on the fly based on the passed parameters
            pass


