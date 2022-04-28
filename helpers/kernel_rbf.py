from helpers.config import Kernel
import numpy as np
from helpers.utils import euclidean_dist_matrix

class RBF(Kernel):
    """
    Radial Basis Function kernel, defined as unnormalized Gaussian PDF
        K(x, y) = e^(-g||x - y||^2)
    where:
        g = gamma
    """

    def __init__(self, gamma=None):
        self._gamma = gamma

    def _compute(self, data_1, data_2):
        if self._gamma is None:
            # libSVM heuristics
            self._gamma = 1./data_1.shape[1]

        dists_sq = euclidean_dist_matrix(data_1, data_2)
        return np.exp(-self._gamma * dists_sq)

    def dim(self):
        return np.inf