from morseg.algorithms.tokenizer import *
from morseg.utils.wrappers import WordlistWrapper


wl = WordlistWrapper.from_file("german.tsv")

for method in LSVTokenizer.param_options["method"]:
    print(method + "\n")
    lsv_model = LSVTokenizer(method=method, strategy="rise")
    try:
        lsv_model.train(wl)
        for f in lsv_model.get_segmentations():
            print(f)
    except:
        print("error :(")

    print("\n" + 100 * "=")

