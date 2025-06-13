import math
from linse.typedsequence import Word, Morpheme
from morseg.utils.wrappers import WordWrapper, WordlistWrapper
from collections import defaultdict

from itertools import combinations
from lingpy.align.pairwise import edit_dist
from lingpy import tokens2class
import matplotlib.pyplot as plt


class KhorsiSimilarity(object):
    def __init__(self, wl : WordlistWrapper):
        self.frequencies = defaultdict(int)
        for w in wl:
            w = w.unsegmented[0]
            for segment in w:
                self.frequencies[segment] += 1

    @staticmethod
    def lcs(X, Y):
        # find the length of the strings
        m = len(X)
        n = len(Y)

        # declaring the array for storing the dp values
        L = [[None] * (n + 1) for i in range(m + 1)]

        """Following steps build L[m + 1][n + 1] in bottom up fashion 
        Note: L[i][j] contains length of LCS of X[0..i-1] 
        and Y[0..j-1]"""
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0:
                    L[i][j] = 0
                elif X[i - 1] == Y[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])

        # backtrack through the matrix
        substring = Morpheme()
        i = m
        j = n

        while i > 0 and j > 0:
            if X[i - 1] == Y[j - 1]:
                substring.insert(0, X[i - 1])
                i -= 1
                j -= 1

            elif X[i - 1] > Y[j - 1]:
                i -= 1
            else:
                j -= 1

        return substring

    def similarity(self, w1, w2):
        w1 = w1.unsegmented[0]
        w2 = w2.unsegmented[0]

        substring = self.lcs(w1, w2)

        # get segments of the two words that are NOT part of the longest common substring
        w1_prime = Morpheme(w1)
        w2_prime = Morpheme(w2)
        for segment in substring:
            w1_prime.remove(segment)
            w2_prime.remove(segment)
        subs_prime = w1_prime + w2_prime

        return (sum([math.log(1 / self.frequencies[segment]) for segment in substring]) -
                sum([math.log(1 / self.frequencies[segment]) for segment in subs_prime]))


if __name__ == "__main__":
    wl = WordlistWrapper.from_file("../../../eval/eval-data/latin-nelex.tsv")
    # wl = WordlistWrapper.from_file("../../../eval/eval-data/lati1261.tsv")
    khorsi = KhorsiSimilarity(wl)

    similarities = []
    ned = []
    sca_dist = []

    for w1, w2 in combinations(wl, 2):
        similarities.append(khorsi.similarity(w1, w2))
        ned.append(edit_dist(w1.unsegmented[0], w2.unsegmented[0], normalized=True))
        sca_dist.append(edit_dist(tokens2class(w1.unsegmented[0], "sca"),
                                  tokens2class(w2.unsegmented[0], "sca"), normalized=True))

    plt.hist(similarities, bins=100)
    plt.show()
    plt.cla()
    plt.hist(ned, bins=25)
    plt.show()
    plt.cla()
    plt.hist(sca_dist, bins=25)
    plt.show()
