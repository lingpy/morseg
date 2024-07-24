"""
Tokenizers are methods that work with pure wordlists.
"""
import morfessor
from typing import List
import random
from collections import defaultdict
from linse.typedsequence import Word
from morseg.utils.wrappers import WordWrapper, WordlistWrapper


def transfer_segmentation(
        from_word: list, 
        to_word: list, 
        from_separators="+",
        to_separator="+"
        ) -> list:
    
    out = [t for t in to_word]
    for i, c in enumerate(from_word):
        if c in from_separators:
            out.insert(i, to_separator)
    return out


# check if entry contains a segment, fast method with strings
def contains(a, b):
    return str(b) in str(a)


def get_word(form):
    return Word([x.split() for x in form.split(" + ")])


# add this to util class later
def merge_pair(word, pair):
    out = get_word(str(word[0]))
    for i in range(1, len(word)):
        seg1, seg2 = word[i - 1], word[i]
        if str(seg1) == str(pair[0]) and str(seg2) == str(pair[1]):
            out[-1].extend(seg2)
        else:
            out.append(seg2)
    return out


def get_vocabulary(words):
    vocabulary = defaultdict(int)
    for word in words:
        vocabulary[word] += 1
    return vocabulary


def get_stats(vocabulary):
    pairs = defaultdict(int)
    for word, freq in vocabulary.items():
        for i in range(len(word) - 1):
            pair = Word.from_string(str(word[i]))
            pair.append(word[i + 1])
            pairs[get_word(
                str(word[i]) + " + " + str(word[i + 1]))
                ] += freq
    return pairs


def merge_vocabulary(pair, vocabulary):
    out = defaultdict(int)
    for word, freq in vocabulary.items():
        if contains(word, pair):
            new_word = merge_pair(word, pair)
            out[new_word] += 1
        else:
            out[word] += 1
    return out


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
        self.forms = words

        # needs to segment all words into individual "morphemes" as a
        # preprocessing step
        segmented_words = []
        for w in words.unsegmented():
            new_word = []
            for morpheme in w:
                new_word += list(morpheme)
            nw = Word(new_word)
            segmented_words += [Word(new_word)]

        self.training_data = segmented_words
    
    def _train(
            self,
            iterations=60,
            threshold=3
            ):
        segmented_words = self.training_data
        
        vocabulary = get_vocabulary(segmented_words)
        for i in range(iterations):
            pairs = get_stats(vocabulary)
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] >= threshold:
                vocabulary = merge_vocabulary(best_pair, vocabulary)
        self.vocabulary = vocabulary
        self.segments = defaultdict(int)
        self.segmented_words = {}
        for word in self.vocabulary:
            unsegmented = Word.from_string(" ".join([str(m) for m in word]))
            self.segmented_words[unsegmented] = word

    def _tokenize(self, word: Word, **kwargs) -> Word:
        """
        Tokenize words into units.

        Note
        ----
        The current strategy just looks up the word in the dictionary, but on
        the long run, we should change strategies for unknown words.
        
        An additional strategy could be to use orthoprofiles to split words by
        the identified segments.
        """
        if word in self.segmented_words:
            return self.segmented_words[word]
        return word


class WordPiece(Tokenizer):
    def _preprocess(self, words: WordlistWrapper, wp_prefix="##"):
        self.training_data = self.forms = words
        self.training_data.split_everywhere()
        self.training_data.add_wp_token(wp_token=wp_prefix)

    def _train(self, iterations=60, threshold=0, wp_prefix="##"):
        alphabet = defaultdict(int)

        for w in self.training_data:
            for m in w:
                alphabet[m] += 1

        for _ in range(iterations):
            # count bigram frequencies
            bigram_freq = defaultdict(int)
            for w in self.training_data:
                for i in range(len(w) - 1):
                    s1 = w[i]
                    s2 = w[i+1]
                    bigram_freq[(s1, s2)] += 1

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

        # store segmented words for retrieval
        # self.segmented_words = {k: v for k, v in zip(self.forms, self.training_data)}


class Morfessor(Tokenizer):
    def _preprocess(self, words: WordlistWrapper):
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
