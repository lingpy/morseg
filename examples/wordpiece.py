import csv
from linse.typedsequence import Word
from morseg.algorithms.tokenizer import WordPiece, PairEncoding


def get_unsegmented_word(wf):
    if isinstance(wf, list):
        wf = " ".join(wf)
    return Word([[x for x in wf.split() if x != "+"]])


# load eval-data
with open("german.tsv") as f:
    words = []
    for row in csv.DictReader(f, delimiter="\t"):
        words += [get_unsegmented_word(row["TOKENS"])]

wp = WordPiece()
wp.train(words, iterations=60)

for w in words:
    print(wp(w))

print("\n" + 100 * "=" + "\n")

bpe = PairEncoding()
bpe.train(words, iterations=60)

for w in words:
    print(bpe(w))
