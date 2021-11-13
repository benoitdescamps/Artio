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