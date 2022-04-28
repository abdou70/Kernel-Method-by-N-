from helpers.config import Kernel
import numpy as np

class Linear(Kernel):
    """
    Linear kernel, defined as a dot product between vectors
        K(x, y) = <x, y>
    """

    def __init__(self):
        self._dim = None

    def _compute(self, data_1, data_2):
        self._dim = data_1.shape[1]
        return data_1.dot(data_2.T)

    def dim(self):
        return self._dim