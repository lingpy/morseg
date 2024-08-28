from morseg.algorithms.tokenizer import *
from morseg.utils.wrappers import WordlistWrapper
from pathlib import Path

FILENAME = "mand1415.tsv"
# FILENAME = "lati1261.tsv"

DATA_DIR = Path(__file__).parent / "eval-data"
RESULTS_DIR = Path(__file__).parent / "eval-results"


def main(model, wl, **kwargs):
    model = model(**kwargs)
    model_name = str(type(model).__name__)

    if type(model) in [LSVTokenizer, LPVTokenizer, LSPVTokenizer]:
        if model.kwargs.get("strategy") == "subword":
            model_name += "-subword"
        else:
            model_name += "-" + model.kwargs.get("method")

    model.train(wl, **kwargs)
    f1, precision, recall = model.forms.f1_score()

    # save metrics to file
    out_file = FILENAME.replace(".tsv", "-metrics.txt")
    with open(RESULTS_DIR / out_file, "a") as f:
        f.write(model_name + "\n")
        f.write(f"f1: {f1}\n")
        f.write(f"precision: {precision}\n")
        f.write(f"recall: {recall}\n")
        f.write(100 * "-" + "\n\n")

    # save segmentations to file
    seg_file = FILENAME.replace(".tsv", f"-{model_name}.txt")
    with open(RESULTS_DIR / seg_file, "w") as f:
        for form in model.get_segmentations():
            f.write(str(form) + "\n")


if __name__ == "__main__":
    filepath = DATA_DIR / FILENAME
    wl = WordlistWrapper.from_file(filepath)

    models = {SquareEntropyTokenizer: dict(),
              LSVTokenizer: {"strategy": "subword"},
              LPVTokenizer: {"strategy": "subword"},
              LSPVTokenizer: {"strategy": "subword"},
              PairEncoding: {"iterations": 20},
              WordPiece: {"iterations": 20},
              Morfessor: dict()
              }

    for model, kwargs in models.items():
        main(model, wl, **kwargs)
        if model in [LSVTokenizer, LPVTokenizer, LSPVTokenizer]:
            for method in ["type", "entropy", "max_drop", "normalized"]:
                kwargs["method"] = method
                kwargs["strategy"] = "peak"
                main(model, wl, **kwargs)
