from abc import ABC, abstractmethod
from typing import List

import numpy as np

from entities.brain import BrainData


class IBrain(ABC):
    """
    This is an interface class to describe the implementation of brain,
    not BrainData.
    """
    @abstractmethod
    def __init__(self,
                 brain_data: BrainData):
        self.brain_data = brain_data

    @abstractmethod
    def get_weights(self) -> BrainData:
        pass

    @abstractmethod
    def mutate_weights(self) -> None:
        pass