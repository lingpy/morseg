from morseg.algorithms.tokenizer import LSVTokenizer, LPVTokenizer, LSPVTokenizer
from morseg.utils.wrappers import WordlistWrapper
from pathlib import Path
import traceback


wl = WordlistWrapper.from_file(Path(__file__).parent.parent / "eval" / "eval-data" / "latin-nelex.tsv")

models = [LSPVTokenizer, LSVTokenizer, LPVTokenizer]

for model in models:
    print(model.__name__)
    for method in LSVTokenizer.param_options["method"]:
        print(method + "\n")
        lsv_model = model(method=method, strategy="peak")
        try:
            lsv_model.train(wl)
            f1, precision, recall = lsv_model.forms.f1_score()
            print(f"F1: {f1}, PRECISION: {precision}, RECALL: {recall}\n")

            for f in lsv_model.get_segmentations():
                print(f)
        except:
            traceback.print_exc()

        print("\n" + 100 * "=")

    lsv_model = model(strategy="subword")
    lsv_model.train(wl)
    print("subword\n")

    f1, precision, recall = lsv_model.forms.f1_score()
    print(f"F1: {f1}, PRECISION: {precision}, RECALL: {recall}\n")

    for f in lsv_model.get_segmentations():
        print(f)

    print("\n\n" + 100 * "+")
