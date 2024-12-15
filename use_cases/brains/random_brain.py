# A function to generate random brain
from typing import List, Tuple

import numpy as np

from entities.creature.brain import BrainData


def generate_brain(layers_sizes: List[int], weights_range: Tuple[float, float]) -> BrainData:
    brain = BrainData(None, None)
    brain.bias_weights = [np.random.uniform(weights_range[0], weights_range[1]) for _ in range(len(layers_sizes)-1)]

    weights = []
    a, b = weights_range
    for i in range(max(len(layers_sizes) - 1, 0)):
        weights.append(
            np.random.uniform(a, b, size=(layers_sizes[i], layers_sizes[i + 1]))
        )

    brain.weights = weights
    return brain
