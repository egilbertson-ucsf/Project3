.. Project 3 documentation master file, created by
   sphinx-quickstart on Fri Mar 19 11:44:38 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to my Neural Net Implementation!
========================================

Neural Network
==============
.. autoclass:: scripts.NN.NeuralNetwork
    :members:

Loss Functions
==============
.. autoclass:: scripts.NN.Loss
   :members:
   :inherited-members:

.. autoclass:: scripts.NN.MSE
   :members:
   :show-inheritance:

.. autoclass:: scripts.NN.CE
   :members:
   :show-inheritance:

Activation Functions
====================

.. autoclass:: scripts.NN.Activation
   :members:
   :inherited-members:

.. autoclass:: scripts.NN.Sigmoid
   :members:
   :show-inheritance:

.. autoclass:: scripts.NN.TanH
   :members:
   :show-inheritance:

.. autoclass:: scripts.NN.Free
   :members:
   :show-inheritance:

File IO
=======

.. autoclass:: scripts.io.FastaReader
   :members:

.. autoclass:: scripts.io.Kmerize
   :members:

Data Transformations and Utilities
==================================

.. autoclass:: scripts.io.norm

.. autoclass:: scripts.io.OneHotEncoding

.. autoclass:: scripts.io.InverseOneHotEncoding

.. autoclass:: scripts.io.TrainTestSplit

.. autoclass:: scripts.io.SubsetData
