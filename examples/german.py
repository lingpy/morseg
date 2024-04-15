from morseg.algorithms.tokenizer import PairEncoding, RandomTokenizer
from linse.typedsequence import Word
import csv

def get_word(wf):
    if isinstance(wf, list):
        wf = " ".join(wf)
    return Word([x.split() for x in wf.split(" + ")])


with open("german.tsv") as f:
    words = []
    words_ = []
    for row in csv.DictReader(f, delimiter="\t"):
        words += [get_word(row["TOKENS"])]
        words_ += [[[x] for x in row["TOKENS"].split(" ")]]

pair = PairEncoding()
pair.train(words)
rtk = RandomTokenizer()
for word in words:
    print(pair(word))
    print(rtk(word))
    print("===")

