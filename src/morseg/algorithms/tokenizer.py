"""
Tokenizers are methods that work with pure wordlists.
"""
import math

from typing import List
import random
from linse.typedsequence import Word, Morpheme
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

    def get_segmentations(self):
        for word in self.forms:
            yield self(word.unsegmented)


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
        "method": ["type", "entropy", "max_drop", "normalized"],
        "strategy": ["peak", "rise", "threshold", "subword"]
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
        self.training_data = Trie(self.forms)

    def _calculate_type_variety(self, token_variety: list):
        return [len(x) for x in token_variety]

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
        return [1 - max(x) / sum(x) for x in token_variety]

    def _calculate_exp_lsv(self):
        # calculate regular LSV first
        type_varieties = [[len(x) for x in token_variety] for token_variety in self.token_varieties.values()]

        # sort variety arrays by their length in descending order
        type_varieties.sort(key=lambda x: len(x), reverse=True)

        self.expected_sv = []

        for i in range(len(type_varieties[0])):
            num_sv = 0
            sum_sv = 0
            for sv in type_varieties:
                if i >= len(sv):
                    break
                num_sv += 1
                sum_sv += sv[i]
            self.expected_sv.append(sum_sv / num_sv)

    def _calculate_norm_lsv(self, token_variety: list):
        """
        Normalized LSV as proposed by Çöltekin (2010).
        """
        # calculate regular LSV first
        norm_sv = self._calculate_type_variety(token_variety)

        # TODO check whether this works properly, oversegmentation is suspiciously strong

        for i in range(len(norm_sv)):
            norm_sv[i] /= self.expected_sv[i]

        return norm_sv

    def _get_token_varieties(self):
        return {word.unsegmented: self.training_data.get_token_variety(word) for word in self.forms}

    def _train(self, **kwargs):
        # cache segmentations with specified parameters
        self.token_varieties = self._get_token_varieties()

        var_func_mapping = {
            "type": self._calculate_type_variety,
            "entropy": self._calculate_successor_entropy,
            "max_drop": self._calculate_successor_max_drop,
            "normalized": self._calculate_norm_lsv
        }

        # get the corresponding function to calculate variety values
        var_func = var_func_mapping.get(self.params["method"], self._calculate_type_variety)

        # preprocessing step for normalized LSV
        if self.params["method"] == "normalized":
            self._calculate_exp_lsv()

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

        for i in range(1, len(varieties)-1):
            if varieties[i] >= varieties[i-1] and varieties[i] >= varieties[i+1]:
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

    def _get_splits_by_subword(self, word):
        subwords = self.training_data.get_subwords(word)
        return [len(x) for x in subwords]

    def _get_splits(self):
        split_func_mapping = {
            "peak": self._get_splits_at_peak,
            "rise": self._get_splits_at_rise,
            "threshold": self._get_splits_by_threshold,
            "subword": self._get_splits_by_subword
        }

        strategy = self.params["strategy"]
        split_func = split_func_mapping.get(strategy, self._get_splits_at_peak)

        splits_by_word = {}

        for word, varieties in self.varieties.items():
            if strategy == "subword":
                splits = split_func(word)
            else:
                splits = split_func(varieties)
            splits_by_word[word] = splits

        return splits_by_word

    def _postprocess(self):
        splits_by_word = self._get_splits()

        for word, splits in splits_by_word.items():
            for i in splits:
                if self.training_data.is_branching(word[0][:i]):
                    self.forms[word].split(i)

    def _tokenize(self, word, **kwargs):
        if (kwargs.get("method", self.params["method"]) == self.params["method"] and
                kwargs.get("strategy", self.params["strategy"]) == self.params["strategy"]):
            return super()._tokenize(word)  # returns the cached segmentation
        else:
            # TODO calculate segmentation on the fly based on the passed parameters
            pass


class LPVTokenizer(LSVTokenizer):
    def _preprocess(self):
        self.training_data = Trie(self.forms, reverse=True)

    def _get_token_varieties(self):
        token_varieties = {}

        for word in self.forms:
            reversed_word = Morpheme(word.unsegmented[0])
            reversed_word.reverse()
            reversed_word = Word(reversed_word)
            token_varieties[reversed_word] = self.training_data.get_token_variety(word)

        return token_varieties

    def _postprocess(self):
        splits_by_word = self._get_splits()

        # word is unsegmented (all segments are in one morpheme)
        for reversed_word, splits in splits_by_word.items():
            word = Morpheme(reversed_word[0])
            word.reverse()
            word_len = len(word)
            word = Word(word)

            for i in splits:
                split_idx = word_len - i
                if self.training_data.is_branching(reversed_word[0][:i]):
                    self.forms[word].split(split_idx)


class LSPVTokenizer(Tokenizer):
    def __init__(self, lsv: LSVTokenizer = None, lpv: LPVTokenizer = None, **kwargs):
        self.lsv = lsv or LSVTokenizer(**kwargs)
        self.lpv = lpv or LPVTokenizer(**kwargs)

        super().__init__(**kwargs)

    def _train(self, **kwargs):
        self.lsv.train(self.forms)
        self.lpv.train(self.forms)

    def _postprocess(self):
        for f in self.forms:
            splits = self.lsv.forms[f.unsegmented].get_splits() + self.lpv.forms[f.unsegmented].get_splits()
            for i in splits:
                f.split(i)
