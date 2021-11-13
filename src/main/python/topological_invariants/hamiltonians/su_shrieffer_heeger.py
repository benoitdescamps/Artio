from topological_invariants.hamiltonians.base import (
ChiralHamiltonian,
SIGMA_X,
SIGMA_Y
)
import numpy as np


class SuSchriefferHeeger(ChiralHamiltonian):

    def __init__(self,alpha:float,beta:float):
        self.alpha = alpha
        self.beta = beta

    def __call__(self,k:float):
        return (self.alpha + self.beta*np.cos(k)) * SIGMA_X \
               + self.beta*np.sin(k) * SIGMA_Y
