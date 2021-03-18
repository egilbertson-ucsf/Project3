#!/usr/bin/env python3
import numpy as np


class NeuralNetwork:

    def __init__(self, layers = [3, 2, 3], learning_rate=0.1):

        self.layers = np.array(layers)
        self.learning_rate = np.array(learning_rate).reshape(1, 1)

        self.params = {
            "weights": [],
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

        for i in np.arange(self.layers.size):

            if i > 0:
                self.params['weights'].append(
                    np.random.random(
                        (self.layers[i], self.layers[i-1])
                    )
                )

                self.params['bias'].append(
                    np.random.random(self.layers[i])
                )



            self.params['zs'].append(
                np.zeros(self.layers[i]).\
                    reshape(self.layers[i], 1)
            )

            self.params['as'].append(
                np.zeros(self.layers[i]).\
                    reshape(self.layers[i], 1)
            )

    def sigmoid(self, x):
        """
        calculates sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    def dSigmoid(self, x):
        sig = self.sigmoid(x)
        return sig - (1 - sig)

    def cost(self, x, y):
        return np.mean((x - y) ** 2)

    def dCost(self, x, y):
        return np.mean(x - y)

    def forward(self, x):
        """
        forward propagation through network
        """

        self.params['as'][0] = x

        for idx in np.arange(1, self.layers.size):

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
            self.params['as'][idx] = self.sigmoid(
                self.params['zs'][idx]
            )

        return self.params['as'][-1]

    def backward(self, y):

        cache_dC_dA_dZ = []
        d_weights = self.params['weights'].copy()
        d_bias = self.params['bias'].copy()

        for idx in np.arange(self.layers.size)[::-1]:

            if idx == 0:
                break

            elif idx == self.layers.size - 1:

                # derivative of cost wrt final activation
                dC_dA = np.full(
                    self.params['as'][idx].size,
                    self.dCost(self.params['as'][idx], y)
                )

            else:

                # calculates current activation derivative using cached layers
                dC_dA = cache_dC_dA_dZ[-1] @ self.params['weights'][idx]


            # derivative of activation wrt z-layer
            dA_dZ = self.dSigmoid(
                self.params['as'][idx]
            )


            # derivative of z-layer wrt to weights
            dZ_dW = self.params['as'][idx-1]

            # perform shared multiplicative term
            dC_dA_dZ = (dC_dA * dA_dZ)

            # derivative of cost wrt to weights
            dC_dW = self.learning_rate * (dC_dA_dZ.T @ dZ_dW)

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
            mean_d_weight_tensor = d_weight_tensor.mean(axis = 0)
            mean_d_bias_tensor = d_bias_tensor.mean(axis = 0)

            # update weights and biases
            self.params['weights'][i] -= mean_d_weight_tensor
            self.params['bias'][i] -= mean_d_bias_tensor

    def clear(self):
        self.d_weights = []
        self.d_bias = []

    def fit(self, x, y, n_epochs = 100, n_iter=50):

        for e in np.arange(n_epochs):

            pred = self.forward(x)
            loss = self.cost(pred, y)

            self.backward(y)

            self.step()
            self.clear()

            if e % 10 == 0:
                print("loss @ epoch {}: {:.4f}".format(e, loss))


def main():
    np.random.seed(42)

    n = 100
    x = np.random.random((1, n))

    nn = NeuralNetwork(
        layers = [n, 5, 5, 5, n]
    )
    nn.fit(x, x)

if __name__ == '__main__':
    main()
