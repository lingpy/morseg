from morseg.algorithms.tokenizer import BytePairTokenizer, RandomTokenizer
from morseg.algorithms.tokenizer import *
from linse.typedsequence import Word


base = ["alle", "jahre", "wieder", "alles", "neu", "macht", "der", "mai", "fallen"]
words = [Word(" + ".join(list(x))) for x in base]

bpt = BytePairTokenizer()

bpt.train(words, iterations=100)
rt = RandomTokenizer()

for word in base:
    print(word)
    print(bpt(Word(" ".join(list(word)))))
    print(rt(Word(" ".join(list(word)))))
    print("===")
