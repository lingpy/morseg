from morseg.algorithms.tokenizer import *
from morseg.utils.wrappers import WordlistWrapper


wl = WordlistWrapper.from_file("german.tsv")

models = [LetterSuccessorVariety, LSVTokenizer, PairEncoding, WordPiece, Morfessor]

for model in models:
    model = model()

    print(type(model).__name__)
    print("\n")

    model.train(wl)
    for f in model.get_segmentations():
        print(f)

    print("\n" + 100 * "=")
