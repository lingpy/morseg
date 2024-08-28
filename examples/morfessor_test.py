import morfessor
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

# can't pass linse objects to Morfessor; but tuples work
train_data = [(1, tuple(str(x).split())) for x in words]

model = morfessor.BaselineModel()
model.load_data(train_data)
model.train_batch()

for s in model.get_segmentations():
    morphemes = [" ".join(x) for x in s[2]]
    print(" + ".join(morphemes))

test_word = "z ɛ ç ts eː n"
test_word = tuple(test_word.split())

print(model.segment(test_word))
