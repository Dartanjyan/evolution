import numpy as np
from use_cases.brains import random_brain

np.set_printoptions(precision=8, suppress=True, formatter={'all': lambda x: f'{x:0.2f}'}, linewidth=100)

layers = [1, 2, 1]

a = random_brain.generate_brain(layers, (-1, 1))

for w in a.bias_weights:
    print(w)
