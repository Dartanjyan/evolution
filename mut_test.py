import numpy as np

a = np.zeros((5, 5), dtype=int)
b = np.ones_like(a, dtype=int)

print("Before mutation")
print(a)
print()
print(b)
print()

rate = 0.5  # in percents

# Создаем маску, где случайные значения больше rate
mask = np.random.random(a.shape) > rate
print(mask)
print()

a = np.where(mask, b, a)

print("After mutation")
print(a)
