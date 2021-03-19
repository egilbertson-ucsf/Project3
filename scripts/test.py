from scripts import NN
from scripts import io

import numpy as np
import os


def test_activation_sigmoid():
    a = NN.Sigmoid()
    test = np.random.random((1, 100))
    test_activation = a.activation(test)
    test_derivative = a.derivative(test)

    assert np.isnan(test_activation).sum() == 0
    assert np.isnan(test_derivative).sum() == 0


def test_activation_tanh():
    a = NN.TanH()
    test = np.random.random((1, 100))
    test_activation = a.activation(test)
    test_derivative = a.derivative(test)

    assert np.isnan(test_activation).sum() == 0
    assert np.isnan(test_derivative).sum() == 0


def test_activation_free():
    a = NN.Free()
    test = np.random.random((1, 100))
    test_activation = a.activation(test)
    test_derivative = a.derivative(test)

    assert np.sum(test - test_activation) == 0
    assert np.unique(test_derivative).size == 1
    assert np.unique(test_derivative)[0] == 1


def test_MSE():
    Loss = NN.MSE()
    test_x = np.random.random((1, 100))
    test_y = np.random.random((1, 100))

    test_cost = Loss.loss(test_x, test_y)
    test_derivative = Loss.derivative(test_x, test_y)

    assert np.isnan(test_cost).sum() == 0
    assert np.isnan(test_derivative).sum() == 0


def test_CE():
    Loss = NN.CE()
    test_x = np.random.random((3, 100))
    test_y = np.random.random((3, 100))

    test_cost = Loss.loss(test_x, test_y)
    test_derivative = Loss.derivative(test_x, test_y)

    assert np.isnan(test_cost).sum() == 0
    assert np.isnan(test_derivative).sum() == 0


def test_fasta_reader():
    path = os.path.abspath("data/yeast-upstream-1k-negative.fa")
    fr = io.FastaReader(path)

    records = np.array([
        header for header, seq in fr
    ])

    assert records.size == 3164


def test_kmer_reader():
    k_size = np.random.choice(np.arange(1, 20))
    path = os.path.abspath("data/yeast-upstream-1k-negative.fa")
    fr = io.FastaReader(path)
    km = io.Kmerize(k=k_size)

    records = []
    for idx, rec in enumerate(fr):
        if idx == 100:
            break
        records.append(rec)

    for kmer in km.process(records, ohe=False):
        assert len(kmer) == k_size

    for ohe in km.process(records, ohe=True):
        assert ohe.size == (k_size * 4)


def test_OHE():
    labels = np.random.choice(np.arange(3), size=(100))

    ohe = io.OneHotEncoding(labels)
    assert ohe.shape == (100, 3)

    ohe_flat = io.OneHotEncoding(labels, flatten=True)
    assert ohe_flat.shape == (300,)

    back_labels = io.InverseOneHotEncoding(
        ohe_flat, [str(i) for i in np.arange(3)]
        )

    assert ''.join([str(i) for i in labels]) == back_labels
