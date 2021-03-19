#!/usr/bin/env python3

import numpy as np


class FastaReader:
    """
    Implementation of a Fasta Reader
    """

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
    """
    Implementation of class to generate kmers from sequences.
    Can interact with any iterable.
    """

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


def InverseOneHotEncoding(ohe, alphabet):
    """
    convert flattened one hot encoding back into original alphabet
    """

    alphabet = np.array(alphabet)

    labels = ohe.\
        reshape(int(ohe.size/alphabet.size), alphabet.size).\
        argmax(axis=1)

    return ''.join(alphabet[labels])


def TrainTestSplit(*args, train_size=0.8):
    """
    shuffles and splits a given set of data evenly
    """

    # confirm all datasets share the same size
    n_obs = np.unique([a.shape[0] for a in args])
    assert n_obs.size == 1

    # identify the size
    n_obs = n_obs[0]

    # calculate the number of observations from the given fraction
    pos = int(n_obs*train_size)

    # initialize the inidices
    indices = np.arange(n_obs)

    # shuffle the indices
    np.random.shuffle(indices)

    # split the random indices into train and test sets
    train_ind = indices[:pos]
    test_ind = indices[pos:]

    # generate the split datasets as [train, test] pairs
    for a in args:
        for i in [train_ind, test_ind]:
            yield a[i]


def SubsetData(data, n=200):
    """
    randomly samples <n> observations from a given set of data w/o replacement
    """
    n = int(n)
    ind = np.arange(data.shape[0])
    np.random.shuffle(ind)
    return data[ind[:n]]
