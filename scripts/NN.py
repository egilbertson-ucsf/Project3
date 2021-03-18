#!/usr/bin/env python3
import numpy as np
from sklearn.datasets import make_blobs
import sys


class Loss:

    def loss(self, *args, **kwargs):
        return self.__loss__(*args, **kwargs)

    def derivative(self, *args, **kwargs):
        return self.__derivative__(*args, **kwargs)


class MSE(Loss):

    def __loss__(self, x, y):
        return np.mean((x - y)**2)

    def __derivative__(self, x, y):
        return np.mean(x - y)


class BCE(Loss):

    def __loss__(self, x, y, eps=1e-12):
        x = np.clip(x, eps, 1-eps)
        y = np.clip(y, eps, 1-eps)
        loss = -np.mean(
            (y * np.log(x)) + ((1-y) * np.log(1-x))
        )

        return loss

    def __derivative__(self, x, y, eps=1e-12):
        x = np.clip(x, eps, 1-eps)
        y = np.clip(y, eps, 1-eps)
        loss = np.mean(
            (x - y) / ((1-x) * x)
        )
        return loss


class Activation:

    def activation(self, *args, **kwargs):
        return self.__activation__(*args, **kwargs)

    def derivative(self, *args, **kwargs):
        return self.__derivative__(*args, **kwargs)


class Sigmoid(Activation):

    def __activation__(self, x):
        """
        calculates sigmoid function
        """

        return 1 / (1 + np.exp(-x))

    def __derivative__(self, x):
        """
        implemented derivative of sigmoid function
        """

        sig = self.activation(x)
        return sig - (1 - sig)


class NeuralNetwork:

    def __init__(self, layers=[3, 2, 3], learning_rate=0.1):

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

            else:

                # calculates current activation derivative using cached layers
                dC_dA = cache_dC_dA_dZ[-1] @ self.params['weights'][idx]

            # derivative of activation wrt z-layer
            dA_dZ = self.params['f'][idx].derivative(
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
            mean_d_weight_tensor = d_weight_tensor.mean(axis=0)
            mean_d_bias_tensor = d_bias_tensor.mean(axis=0)

            # update weights and biases
            self.params['weights'][i] -= mean_d_weight_tensor
            self.params['bias'][i] -= mean_d_bias_tensor

    def clear(self):
        self.d_weights = []
        self.d_bias = []

    def fit(self, x, y, Loss, n_epochs=100, n_iter=50):

        for e in np.arange(n_epochs):

            pred = self.forward(x)
            loss = Loss.loss(pred, y)

            self.backward(y, Loss)

            self.step()
            self.clear()

            if e % 10 == 0:
                print("loss @ epoch {}: {:.4f}".format(e, loss))


def main():

    np.random.seed(42)

    # X, Y = make_blobs(n_samples=100, n_features=4, centers=2)
    # Y = Y.reshape(Y.size, 1)
    # print(Y)

    n = 4
    x = np.random.random((1, n))
    y = x.copy()

    nn = NeuralNetwork(
        layers=[
            (4, None),
            (3, Sigmoid),
            (4, Sigmoid)
        ],
        learning_rate=0.01
    )


    loss = MSE()
    # nn.forward(x)
    # nn.backward(x, loss)
    nn.fit(x, y, loss, n_epochs=100)
    # loss_fn = BCE()

    # for epoch in np.arange(1000):
    #     losses = []
    #     predictions = []
    #     for x, y in zip(X, Y):
    #         pred = nn.forward(x)
    #         loss = loss_fn.loss(pred, y)
    #
    #         nn.backward(y, loss_fn)
    #
    #         predictions.append(pred)
    #         losses.append(loss)
    #
    #     nn.step()
    #     print("Mean Loss at epoch {} : {:.6f}".format(epoch, np.mean(losses)))
    #
    #     if epoch == 5:
    #         print(np.array(predictions).ravel())
    #         print(Y.ravel())
    #         break


if __name__ == '__main__':
    main()
