from topological_invariants.hamiltonians.base import (
ChiralHamiltonian,
SIGMA_X,
SIGMA_Y
)
import numpy as np
from typing import List


class General1DChiralHamiltonian(ChiralHamiltonian):
    """
    According to Machine Learning Topological Invariants with Neural Networks,
    bit doubtful this is the most general though 1D case though
    """

    def __init__(self,
                 n:int,
                 alpha: List[float],
                 beta: List[float]):
        self.n = n
        self.alpha = alpha
        self.beta = beta

    def __call__(self, k: float):
        hx = self.alpha*np.cos(np.arange(self.n)*k)* SIGMA_X
        hy = self.beta*np.sin(np.arange(self.n)*k)* SIGMA_Y
        return hx + hy
