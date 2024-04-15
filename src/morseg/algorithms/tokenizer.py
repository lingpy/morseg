"""
Tokenizers are methods that work with pure wordlists.
"""
from typing import List
import random
from collections import defaultdict
from linse.typedsequence import Word


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

    def _train(self, words, **kwargs):
        pass

    def train(
            self, 
            words: List[Word], 
            **kwargs):
        self.training_data = words
        self._train(self.training_data, **kwargs)
    
    def _tokenize(self, word, **kwargs):
        return word

    def __call__(self, word: Word, **kwargs) -> Word:
        return self._tokenize(word, **kwargs)

    def tokenize(
            self, 
            words: List[Word], 
            **kwargs
            ):
        for word in words:
            yield self(word, **kwargs)


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
    
    def _train(
            self, 
            words: List[Word], 
            iterations=100, 
            threshold=3
            ):
        # needs to segment all words into individual "morphemes" as a
        # preprocessing step
        self.words = words
        segmented_words = []
        for w in words:
            new_word = []
            for morpheme in w:
                new_word += list(morpheme)
            nw = Word(new_word)
            segmented_words += [Word(new_word)]
        
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

