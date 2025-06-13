# Morpheme Segmentation in Multi- and Monolingual Wordlists 

This package provides implementations for several algorithms by which words in a wordlist can be segmented into morphemes.

If you use this software package, please cite it accordingly:

> Rubehn, A. and J.-M. List (2025). MorSeg: A Python package for morpheme segmentation in multi- and monolingual wordlists [Software Library, Version 0.1]. Chair for Multilingual Computational Linguistics, University of Passau.


## Installation

This package can be conveniently installed using pip:

```
pip install morseg
```

## Basic Usage

### Loading data

Assuming your data is presented in a TSV file following the LingPy specifications (see `/tests/test_data/german.tsv` for an example), you can simply load your data with:

```python
from morseg.utils.wrappers import WordlistWrapper

wl = WordlistWrapper.from_file(YOUR_FILE)
```

This creates a wordlist wrapper object; a representation of a wordlist with three annotation levels: The predicted segmentations (by a model), the Gold standard segmentations, and the unsegmented form. The training of all models requires the data to be stored in this class!

### Training a model

The `Tokenizer` class offers a unified interface for all models that are implemented in this library. For example, if you want to train a LSV (Letter Successor Variety) model, you can simply do so like that:

```python
from morseg.algorithms.tokenizer import LSVTokenizer

model = LSVTokenizer()
model.train(wl)
```

The current release covers implementations of the following models:

- `LSVTokenizer`: Letter Successor Variety ([Harris, 1955](https://doi.org/10.2307/411036)) with the following adaptations:
  - Letter Successor Entropy ([Hafer and Weiss, 1974](https://doi.org/10.1016/0020-0271(74)90044-8))
  - Letter Max-Drop Variety ([Hammarström, 2009](https://gupea.ub.gu.se/handle/2077/21418?show=full))
  - Normalized Letter Successor Variety ([Çöltekin, 2010](https://coltekin.net/cagri/papers/coltekin2010clin.pdf))
- `LPVTokenizer`: Letter Predecessor Variety (analogically to LSV, but processing the words backwards)
- `LSPVTokenizer`: A combination of Letter Successor Variety and Letter Predecessor Variety
- `Morfessor`: The Morfessor Baseline Model ([Creutz and Lagus, 2002](https://doi.org/10.3115/1118647.1118650))
- `SquareEntropyTokenizer` ([Méndez-Cruz et al., 2016](https://doi.org/10.1016/j.patrec.2016.09.001))

Furthermore, some popular models for subword tokenization are implemented:
- `PairEncoding`: Byte-Pair Encoding ([Sennrich et al., 2016](https://10.18653/v1/P16-1162))
- `WordPiece` ([Schuster and Nakajima, 2012](https://doi.org/10.1109/ICASSP.2012.6289079))
- `UnigramSentencePiece` ([Kudo, 2018](https://doi.org/10.18653/v1/P18-1007))

### Obtain segmentations

You can obtain the predicted segmentations from your training data by calling:

```python
for segmented_word in model.get_segmentations():
    # do whatever
```

You can also try segmenting unseen words (depending on the model, this might work more or less well):

```python
word = ["w", "o", "r", "d"]
segmented_word = model(word)
```



