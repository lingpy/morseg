import linguistica
import linguistica as lxa
import csv
from linse.typedsequence import Word


def get_word(wf):
    if isinstance(wf, list):
        wf = " ".join(wf)
    return Word([x.split() for x in wf.split(" + ")])


# load eval-data
with open("german.tsv") as f:
    words = []
    for row in csv.DictReader(f, delimiter="\t"):
        words += [get_word(row["TOKENS"])]


linguistica.from_wordlist(words)
