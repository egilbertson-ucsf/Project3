#!/usr/bin/env python3

import numpy as np


class FastaReader:

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        """
        reads a fasta formatted file
        """

        with open(self.path, "r") as f:
            header = None
            seq = ""
            while True:
                try:
                    line = next(f)
                    if ">" in line:
                        if not header:
                            header = line.strip()
                            seq = ""
                            continue
                        else:
                            rec = (header, seq)
                            header = line.strip()
                            seq = ""
                            yield rec
                    else:
                        seq += line.strip()

                except StopIteration:
                    break


class Kmerize:
    def __init__(self, k=17):
        self.k = k
        self.lookup = {
            "A": 0,
            "C": 1,
            "T": 2,
            "G": 3
        }

    def get_kmers(self, s):
        """
        calculates kmers as slices of sequences
        """

        for i in range(len(s) - self.k + 1):
            yield s[i:i+self.k]

    def process(self, iterable, ohe=False):
        """
        iterates through sequences and returns kmers

        Optionally can return One Hot Encoded Kmers
        """
        for header, seq in iterable:
            for kmers in self.get_kmers(seq):

                if ohe:
                    yield OneHotEncoding(kmers, self.lookup, flatten=True)

                else:
                    yield kmers


def norm(X):
    """
    Applies a normalization transformation to a given dataset

    1) Shifts all values to positive range (+abs(min))
    2) Scale values to 1/max in column
    """

    norm = X.copy()
    for i in np.arange(X.shape[1]):
        norm[:, i] += np.abs(np.min(X[:, i]))
        norm[:, i] /= np.max(norm[:, i])

    return norm


def OneHotEncoding(labels, lookup=None, flatten=False):
    """
    transforms a given vector of labels into a one hot encoded matrix
    """
    if not lookup:

        classes = np.unique(labels)
        n = classes.size
        lookup = {
            c: i for i, c in enumerate(classes)
        }

    else:
        n = len(lookup)

    ohe = np.zeros((len(labels), n))
    for idx, l in enumerate(labels):
        ohe[idx, lookup[l]] = 1

    if flatten:
        return ohe.ravel()

    return ohe


def main():

    fa_path = "../data/yeast-upstream-1k-negative.fa"

    fa = FastaReader(fa_path)
    km = Kmerize()

    for kmer in km.process(fa, ohe=True):
        print(kmer)


if __name__ == '__main__':
    main()
