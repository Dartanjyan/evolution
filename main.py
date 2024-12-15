import numpy as np

from entities.brain import BrainData

np.set_printoptions(precision=8, suppress=True, formatter={'all': lambda x: f'{x:0.2f}'}, linewidth=100)

layers = [3, 3, 1]
a = BrainData([np.random.uniform(-1, 1, (layers[i], layers[i+1])) for i in range(len(layers)-1)],
              [np.random.uniform(-3, 3) for _ in range(len(layers)-1)])

print(a.weights)
