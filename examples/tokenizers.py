from morseg.algorithms.tokenizer import *
from morseg.utils.wrappers import WordlistWrapper


wl = WordlistWrapper.from_file("latin-nelex.tsv")

models = [SquareEntropyTokenizer, LSVTokenizer, PairEncoding, WordPiece, Morfessor]

for model in models:
    model = model()

    print(type(model).__name__)
    print("\n")

    model.train(wl)

    f1, precision, recall = model.forms.f1_score()
    print(f"F1: {f1}, PRECISION: {precision}, RECALL: {recall}\n")

    for f in model.get_segmentations():
        print(f)

    print("\n" + 100 * "=")
