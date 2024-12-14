from abc import ABC, abstractmethod
from typing import List

import numpy as np


class IBrain(ABC):
    @abstractmethod
    def get_weights(self) -> List[np.ndarray]:
        pass

    @abstractmethod
    def mutate_weights(self):
        pass