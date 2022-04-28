from helpers.config import Kernel
import numpy as np

class Polynomial(Kernel):
    """
    Polynomial kernel, defined as a power of an affine transformation
        K(x, y) = (a<x, y> + b)^p
    where:
        a = scale
        b = bias
        p = degree
    """

    def __init__(self, scale=1, bias=0, degree=2):
        self._dim = None
        self._scale = scale
        self._bias = bias
        self._degree = degree

    def _compute(self, data_1, data_2):
        self._dim = data_1.shape[1]
        return (self._scale * data_1.dot(data_2.T) + self._bias) ** self._degree

    def dim(self):
        return self._dim ** self._degree
