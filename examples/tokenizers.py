from morseg.algorithms.tokenizer import *
from morseg.utils.wrappers import WordlistWrapper
import matplotlib.pyplot as plt


wl = WordlistWrapper.from_file("latin-nelex.tsv")

# models = [SquareEntropyTokenizer, LSVTokenizer, LPVTokenizer, LSPVTokenizer, PairEncoding, WordPiece, Morfessor]
models = [PairEncoding, WordPiece]

for model in models:
    model = model()
    model_name = str(type(model).__name__)

    print(model_name)
    print("\n")

    model.train(wl, iterations=10000, threshold=0, callbacks=["alphabet_size", "f1"])

    fig, ax1 = plt.subplots()

    plt.title(model_name)

    color = "tab:blue"
    ax1.set_xlabel("iterations")
    ax1.set_ylabel("alphabet size", color=color)
    ax1.plot(model.training_history["alphabet_size"], color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("F1 score", color=color)
    ax2.plot(model.training_history["f1"], color=color)
    ax2.plot(model.training_history["precision"], "g--", label="precision")
    ax2.plot(model.training_history["recall"], "y--", label="recall")
    ax2.tick_params(axis="y", labelcolor=color)

    ax2.legend()

    fig.tight_layout()
    plt.show()
    plt.cla()

    f1, precision, recall = model.forms.f1_score()
    print(f"F1: {f1}, PRECISION: {precision}, RECALL: {recall}\n")

    # for f in model.get_segmentations():
    #    print(f)

    print("\n" + 100 * "=")
