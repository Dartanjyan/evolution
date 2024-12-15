from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class BrainData:
    weights: Optional[List[np.ndarray | List[float]]]
    bias_weights: Optional[List[float]]
