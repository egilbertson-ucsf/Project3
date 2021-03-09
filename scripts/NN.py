#!/usr/bin/env python3

import numpy as np


class NeuralNetwork:

    def __init__(self, layers, learning_rate=0.01):
        self.layers_config = np.array(layers)
        self.learning_rate = learning_rate
        self.weights, self.bias = self.make_weights()
        self.nodes = [np.zeros(x) for x in self.layers_config]
        self.activations = [np.zeros_like(x) for x in self.nodes]

    def make_weights(self):
        """
        generates initial weights in list

        each weight matrix (N_l x N_l+1) represented the edges between
        all nodes in layer l to all nodes in layer L+1.
        """

        self.weights = []
        self.bias = []
        for i in np.arange(self.layers_config.size-1):
            weights = np.random.random(
                (self.layers_config[i], self.layers_config[i + 1])
                )
            bias = np.random.random(self.layers_config[i + 1])
            self.weights.append(weights)
            self.bias.append(bias)

        return self.weights, self.bias

    def Sigmoid(self, x):
        """
        calculates sigmoid function
        """

        return 1 / (1 + np.exp(-x))

    def dSigmoid(self, a):
        """
        calculates derivative of precalculated sigmoid function
        """
        # sig = self.Sigmoid(x)
        return a - (1.0 - a)

    def MSE(self, x, y):
        """
        calculates mean squared error
        """
        assert x.size == y.size

        return np.sum((x - y)**2) / x.size

    def dMSE(self, x, y):
        """
        implements partial derivative of MSE w.r.t. x
        """
        assert x.size == y.size

        return (2 / x.size) * np.sum(x-y)

    def feedforward(self, x):
        """
        feeds input data through forward passes
        implemented with dot matrix multiplication of
        previous layer nodes and inter-layer weights
        """
        self.nodes = [x] + [np.zeros(x) for x in self.layers_config[1:]]
        self.activations = [np.zeros_like(x) for x in self.nodes]

        for i in np.arange(self.layers_config.size-1):
            self.nodes[i + 1] = \
                (self.activations[i] @ self.weights[i]) + self.bias[i]

            self.activations[i+1] = self.Sigmoid(self.nodes[i+1])

    def backprop(self, y):

        for ldx in np.arange(1, self.layers_config.size)[::-1]:

            if ldx == self.layers_config.size-1:

                # calculates initial loss
                loss = self.MSE(self.activations[ldx], y)

                # deriv of Loss w.r.t Activation
                dL_dA = self.dMSE(self.activations[ldx], y)

            else:

                # calculates loss from upstream neurons
                loss = self.MSE(self.activations[ldx], self.activations[ldx+1])

                # deriv of Loss w.r.t Activation
                dL_dA = self.dMSE(self.activations[ldx], self.activations[ldx+1])

            # deriv of Activation w.r.t Node
            dA_dZ = self.dSigmoid(self.activations[ldx])

            # deriv of Node w.r.t Weight
            dZ_dW = self.activations[ldx-1]

            # calculate partial deriviatives as product of previous terms
            dL_dW = np.zeros_like(self.weights[ldx-1])
            for a in np.arange(dA_dZ.size):
                for w in np.arange(dZ_dW.size):
                    dL_dW[w, a] = (
                        dL_dA * dA_dZ[a] * dZ_dW[w]
                    )

            # update weights in layer
            self.weights[ldx-1] = self.weights[ldx-1] - (self.learning_rate * dL_dW)

            print(self.weights)

    def fit(self):
        pass

    def predict(self):
        pass


def activation(x):
    pass


def main():
    np.random.seed(42)

    nn = NeuralNetwork(
        layers=[8, 3, 8]
    )

    input_data = np.random.random(8)
    nn.feedforward(input_data)
    nn.backprop(input_data)


if __name__ == '__main__':
    main()
