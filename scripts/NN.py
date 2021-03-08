#!/usr/bin/env python3

import numpy as np


class NeuralNetwork:

    def __init__(self, layers):
        self.layers_config = np.array(layers)
        self.weights = self.make_weights()
        self.nodes = [np.zeros(x) for x in self.layers_config]

    def make_weights(self):
        """
        generates initial weights in list

        each weight matrix (N_l x N_l+1) represented the edges between
        all nodes in layer l to all nodes in layer L+1.
        """

        self.weights = []
        for i in np.arange(self.layers_config.size-1):
            weights = np.random.random(
                (self.layers_config[i], self.layers_config[i + 1])
                )
            self.weights.append(weights)

        return self.weights

    def feedforward(self, x):
        """
        feeds input data through forward passes
        implemented with dot matrix multiplication of
        previous layer nodes and inter-layer weights
        """
        self.nodes = [x] + [np.zeros(x) for x in self.layers_config[1:]]
        for i in np.arange(self.layers_config.size-1):
            weights = self.weights[i]
            self.nodes[i + 1] = self.nodes[i] @ weights

    def backprop(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


def activation(x):
    pass


def main():

    nn = NeuralNetwork(
        layers=[8, 3, 8]
    )

    input_data = np.random.random(8)
    nn.feedforward(input_data)


if __name__ == '__main__':
    main()
