import numpy as np

from entities.brain import BrainData

layers = [3, 3, 1]
a = BrainData([np.random.uniform(-1, 1, (layers[i], layers[i+1])) for i in range(len(layers)-1)],
              [np.random.uniform(-3, 3) for _ in range(len(layers)-1)])

print(a)
