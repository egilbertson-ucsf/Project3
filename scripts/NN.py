#!/usr/bin/env python3
import numpy as np
from sklearn.datasets import make_blobs
import sys


class Loss:
    """
    Parent Class for Loss Functions
    """

    def loss(self, *args, **kwargs):
        return self.__loss__(*args, **kwargs)

    def derivative(self, *args, **kwargs):
        return self.__derivative__(*args, **kwargs)


class MSE(Loss):

    def __loss__(self, x, y):
        """
        Calculates Mean Squared Error Loss Function
        """

        return np.mean((x - y)**2)

    def __derivative__(self, x, y):
        """
        Calculates Derivative of Mean Squared Error w.r.t to X
        """

        return np.mean(x - y)


class CE(Loss):
    """
    Cross Entropy Loss with SoftMax included
    """

    def softmax(self, x):
        """
        calculates softmax of a given array
        """

        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def __loss__(self, x, y):
        """
        Applies softmax to an array then calculates loss
        """

        p = self.softmax(x)
        loss = np.sum(-y * np.log(p))
        return loss

    def __derivative__(self, x, y):
        """
        Calculates derivative of Cross Entropy w.r.t X
        (SoftMax gradient included)
        """

        grad = self.softmax(x) - y
        return grad


class Activation:
    """
    Parent class for Activation Functions
    """

    def activation(self, *args, **kwargs):
        return self.__activation__(*args, **kwargs)

    def derivative(self, *args, **kwargs):
        return self.__derivative__(*args, **kwargs)


class Sigmoid(Activation):
    """
    Sigmoidal Activation function

    AKA the zero to one squishification.
    """

    def __positive__(self, x):
        """
        calculates positive terms
        """

        return 1 / (1 + np.exp(-x))

    def __negative__(self, x):
        """
        calculates negative terms
        """

        exps = np.exp(x)
        return exps / (1 + exps)

    def __activation__(self, x):
        """
        calculates sigmoid function in a numerically stable way
        """

        mask_p = x >= 0
        mask_n = ~mask_p

        sig = np.zeros_like(x)
        if np.any(mask_p):
            sig[mask_p] = self.__positive__(x[mask_p])

        if np.any(mask_n):
            sig[mask_n] = self.__negative__(x[mask_n])

        return sig

    def __derivative__(self, x):
        """
        implemented derivative of sigmoid function
        """

        sig = self.activation(x)
        return sig - (1 - sig)


class Free(Activation):
    """
    Empty Activation Layer
    """

    def __activation__(self, x):
        """
        pass current input forward without modification
        """
        return x

    def __derivative__(self, x):
        """
        pass constant derivative backwards
        """

        return 1


class NeuralNetwork:

    def __init__(self, layers, learning_rate=0.1):

        self.layers = np.array(layers)
        self.learning_rate = np.array(learning_rate).reshape(1, 1)

        self.params = {
            "weights": [],
            "f": [],
            "bias": [],
            "zs": [],
            "as": []
        }

        self.d_weights = []
        self.d_bias = []

        self.initialize_params()

    def initialize_params(self):
        """
        randomly initializes weights and biases

        stores all internal parameters within indexable dictionary
        """

        for i in np.arange(self.layers.shape[0]):

            if i > 0:
                self.params['weights'].append(
                    np.random.random(
                        (self.layers[i][0], self.layers[i-1][0])
                    )
                )

                self.params['bias'].append(
                    np.random.random(self.layers[i][0])
                )

            self.params['zs'].append(
                np.zeros(self.layers[i][0]).reshape(self.layers[i][0], 1)
            )

            self.params['as'].append(
                np.zeros(self.layers[i][0]).reshape(self.layers[i][0], 1)
            )

            if self.layers[i][1] is None:
                self.params['f'].append(
                    self.layers[i][1]
                )

            else:
                self.params['f'].append(
                    self.layers[i][1]()
                )

    def forward(self, x):
        """
        forward propagation through network
        """

        self.params['as'][0] = x

        for idx in np.arange(1, self.layers.shape[0]):

            # index previous activations
            a = self.params['as'][idx-1]

            # index weights
            w = self.params['weights'][idx-1]

            # index bias
            b = self.params['bias'][idx-1]

            # calculate z_array
            self.params['zs'][idx] = (
                (a @ w.T) + b
            )

            # calculate activation
            self.params['as'][idx] = self.params['f'][idx].activation(
                self.params['zs'][idx]
            )

        return self.params['as'][-1]

    def backward(self, y, loss):
        """
        calculates gradients via backpropagation
        """

        cache_dC_dA_dZ = []
        d_weights = self.params['weights'].copy()
        d_bias = self.params['bias'].copy()

        for idx in np.arange(self.layers.shape[0])[::-1]:
            if idx == 0:
                break

            elif idx == self.layers.shape[0] - 1:

                # derivative of cost wrt final activation
                dC_dA = np.full(
                    self.layers[idx][0],
                    loss.derivative(self.params['as'][idx], y)
                )
                dC_dA = dC_dA.reshape(1, dC_dA.size)

            else:

                # calculates current activation derivative using cached layers
                dC_dA = cache_dC_dA_dZ[-1].T @ self.params['weights'][idx]

            # derivative of activation wrt z-layer
            dA_dZ = self.params['f'][idx].derivative(
                self.params['as'][idx]
            )

            # derivative of z-layer wrt to weights
            dZ_dW = self.params['as'][idx-1].\
                reshape(1, self.params['as'][idx-1].size)

            # perform shared multiplicative term
            dC_dA_dZ = (dC_dA * dA_dZ).reshape(self.layers[idx][0], 1)

            # derivative of cost wrt to weights
            dC_dW = self.learning_rate * (dC_dA_dZ @ dZ_dW)

            # derivative of cost wrt to bias
            dC_dB = (self.learning_rate * dC_dA_dZ).reshape(-1)

            # cache calculated derivatives
            cache_dC_dA_dZ.append(dC_dA_dZ)
            d_weights[idx - 1] = dC_dW.copy()
            d_bias[idx - 1] = dC_dB.copy()

        self.d_weights.append(d_weights)
        self.d_bias.append(d_bias)

    def step(self):

        # iterate through layers
        for i in np.arange(len(self.params['weights'])):

            # prepare tensor
            d_weight_tensor = []
            d_bias_tensor = []

            # iterate through all derivatives
            for mdx in np.arange(len(self.d_weights)):

                dw = self.d_weights[mdx][i]
                db = self.d_bias[mdx][i]

                d_weight_tensor.append(dw)
                d_bias_tensor.append(db)

            # consolidate arrays into multidimensional tensor
            d_weight_tensor = np.array(d_weight_tensor)
            d_bias_tensor = np.array(d_bias_tensor)

            # take mean across the zero axis
            mean_d_weight_tensor = d_weight_tensor.mean(axis=0)
            mean_d_bias_tensor = d_bias_tensor.mean(axis=0)

            # update weights and biases
            self.params['weights'][i] -= mean_d_weight_tensor
            self.params['bias'][i] -= mean_d_bias_tensor

    def clear(self):
        self.d_weights = []
        self.d_bias = []

    def fit(self, X, Y, Loss, n_epochs=100):
        """
        trains model given X:observations and Y:labels
        """

        for epoch in np.arange(n_epochs):
            losses = []
            for x, y in zip(X, Y):
                pred = self.forward(x)
                loss = Loss.loss(pred, y)
                self.backward(y, Loss)

                losses.append(loss)

            self.step()
            self.clear()

            if epoch % 10 == 0:
                print(
                    "Mean Loss at epoch {} : {:.6f}".format(
                        epoch, np.mean(losses)
                        )
                )

    def predict(self, X):
        """
        feeds forward all observations in X and returns predictions
        """

        predictions = []
        for x in X:
            pred = self.forward(x)
            predictions.append(pred)

        return np.array(predictions)


def main():

    np.random.seed(42)

    X, labels = make_blobs(n_samples=100, n_features=8, centers=3)
    Y = np.zeros((labels.size, 4))
    Y[np.flatnonzero(labels == 0), 0] = 1
    Y[np.flatnonzero(labels == 1), 1] = 1
    Y[np.flatnonzero(labels == 2), 2] = 1
    Y[np.flatnonzero(labels == 3), 3] = 1

    nn = NeuralNetwork(
        layers=[
            (8, None),
            (4, Sigmoid),
            (4, Free)
        ],
        learning_rate=0.05
    )
    # Loss = MSE()
    Loss = CE()

    nn.fit(X, Y, Loss, n_epochs=300)
    nn.predict(X)

if __name__ == '__main__':
    main()
