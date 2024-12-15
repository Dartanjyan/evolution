from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BrainData:
    weights: List[np.ndarray]
    bias_weights: List[float]
