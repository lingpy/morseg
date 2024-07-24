from morseg.algorithms.tokenizer import PairEncoding
from morseg.algorithms.tokenizer import *
from linse.typedsequence import Word

class TT(tuple):
    
    def __new__(self, seq):
        sequence = []
        for m in seq.split(" + "):
            sequence += [tuple(m.split())]
        return tuple.__new__(TT, sequence)

    def __str__(self):
        return " + ".join([" ".join(m) for m in self])

    def __eq__(self, other):
        return str(self) == str(other)

    def __hash__(self):
        return hash(str(self))




t = TT("h au s + m ai s")
t2 = TT("a + b + c")
t3 = TT("b + c")

def merge_pair(word, pair):
    out = Word(word.morphemes[0])
    for i in range(1, len(word.morphemes)):
        seg1, seg2 = word.morphemes[i - 1], word.morphemes[i]
        if seg1 == pair.morphemes[0] and seg2 == pair.morphemes[1]:
            out.append(seg2)
        else:
            out.extend(seg2)
    return out

# x = merge_pair(Word("a + b + c"), Word("a + b"))
# print(x)
#
base = ["alle", "jahre", "wieder", "alles", "neu", "macht", "der", "mai", "fallen"]
words = [Word(" + ".join(list(x))) for x in base]

bpe = PairEncoding()

bpe.train(words, iterations=100)

for word in base:
    print(bpe(" ".join(list(word))))

