import numpy as np


def sigmoid(Z):
    """
    It returns sigmoid of Z and Z without modification which is used as cache for bacward propagation.
    """
    return 1 / (1 + np.exp(-Z)), Z


def sigmoid_backward(dA, cache):
    """
    It accepts dA, i.e gradient of activation and output Z as cache to calculate dZ, i.e gradient of Z.
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def relu(Z):
    """
    Implementation of ReLU(Rectified Linear Unit).
    """
    return np.maximum(0, Z), Z


def relu_backward(dA, cache):
    """
    Relu for backward propagation.
    """
    Z = cache
    dZ = np.array(dA, copy=True)

    dZ[Z <= 0] = 0

    return dZ
