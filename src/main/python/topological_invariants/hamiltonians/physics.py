from abc import ABC, abstractmethod
import numpy as np
from typing import List

SIGMA_X = np.array([[0,1],[1,0]])
SIGMA_Y = np.array([[0,1],[-1,0]])


class ChiralHamiltonian(ABC):
    """
    Interface for physics models to
    generate
    H = hx sigmaX + hy sigmaY
    """

    @abstractmethod
    def __call__(self,k:float):
        pass


class SuSchriefferHeeger(ChiralHamiltonian):

    def __init__(self,alpha:float,beta:float):
        self.alpha = alpha
        self.beta = beta

    def __call__(self,k:float):
        return (self.alpha + self.beta*np.cos(k)) * SIGMA_X \
               + self.beta*np.sin(k) * SIGMA_Y


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

