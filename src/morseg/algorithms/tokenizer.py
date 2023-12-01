"""
Tokenizers are methods that work with pure wordlists.
"""
import random
from collections import defaultdict
from linse.typedsequence import Word

# check if entry contains a segment, fast method with strings
def contains(a, b):
    return str(b) in str(a)


# add this to util class later
def merge_pair(word, pair):
    out = Word(word.morphemes[0])
    for i in range(1, len(word.morphemes)):
        seg1, seg2 = word.morphemes[i - 1], word.morphemes[i]
        if seg1 == pair.morphemes[0] and seg2 == pair.morphemes[1]:
            out.append(seg2)
        else:
            out.extend(seg2)
    return out #Word(str(out))


def get_vocabulary(words):
    vocabulary = defaultdict(int)
    for word in words:
        vocabulary[word] += 1
    return vocabulary


def get_stats(vocabulary):
    pairs = defaultdict(int)
    for word, freq in vocabulary.items():
        for i in range(len(word.morphemes) - 1):
            pair = Word(word.morphemes[i])
            pair.extend(word.morphemes[i + 1])
            pairs[Word(
                str(word.morphemes[i] + " + " + str(word.morphemes[i + 1])))
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

    def train(self, words, **kwargs):
        self.training_data = words
        self._train(self.training_data, **kwargs)
    
    def _tokenize(self, word, **kwargs):
        return word

    def __call__(self, word, **kwargs):
        return self._tokenize(word, **kwargs)

    def tokenize(self, words, **kwargs):
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

    def _tokenize(self, word, **kwargs):
        # get number of break points 
        idxs = list(range(len(word)))
        break_point_number = random.randint(
                0, 
                int((len(word) - 2) * self.kwargs["morpheme_ratio"] + 0.5))
        break_points = random.sample(idxs[1:-1], break_point_number)
        out = Word(word[0])
        for i in range(1, len(word)):
            if i in break_points:
                out.extend(word[i])
            else:
                out.append(word[i])
        return out


class BytePairTokenizer(Tokenizer):
    """

    Notes
    -----
    Code taken with modifications from https://www.geeksforgeeks.org/byte-pair-encoding-bpe-in-nlp/
    """
    def __init__(self):
        Tokenizer.__init__(self)
    
    def _train(self, words, iterations=100, threshold=2):
        vocabulary = get_vocabulary(words)
        for i in range(iterations):
            pairs = get_stats(vocabulary)
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] >= threshold:
                vocabulary = merge_vocabulary(best_pair, vocabulary)
        self.vocabulary = vocabulary
        self.segments = defaultdict(int)
        self.segmented_words = {}
        for word in self.vocabulary:
            unsegmented = Word(" ".join([str(m) for m in word.morphemes]))
            self.segmented_words[unsegmented] = word

    def _tokenize(self, word, **kwargs):
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

