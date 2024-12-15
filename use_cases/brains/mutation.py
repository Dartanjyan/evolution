#
#   Алгоритм 0: скопировать мозг наилучшего существа (наибольшая координата <x>),
# пройтись по каждому весу и с шансом self.chance_to_mutate изменить на случайное число из
# промежутка self.mutation_range, а также добавить лучшее существо из прошлого поколения
#
#   Алгоритм 1: скопировать мозг наилучшего существа (наибольшая координата <x>),
# и определённое количество случайных весов изменить на случайное число из промежутка
# self.mutation_range, а также добавить лучшее существо из прошлого поколения + 3 новых
#
#   Алгоритм 2: скопировать мозги нескольких лучших существ (наибольшая координата <x>).
# Новый мозг сначала - это копия лучшего родителя. Потом по каждому весу пройтись и с шансом 50%
# заменить его весом другого родителя по этому же адресу. Затем с шансом self.chance_to_mutate изменить
# его на случайное число из промежутка self.mutation_range.
from copy import deepcopy
from typing import Tuple

import numpy as np

from entities.brain import BrainData


def mutate_with_chance(brain: BrainData, chance_to_mutate: float, mutation_range: Tuple[float, float]) -> BrainData:
    # С шансом каждый вес изменить
    mutated_brain = deepcopy(brain)  # Создаем копию
    for i, layer in enumerate(mutated_brain.weights):
        for j in range(len(layer)):
            if np.random.random() < chance_to_mutate:
                layer[j] += np.random.uniform(mutation_range[0], mutation_range[1])
    # Обрабатываем bias_weights
    for k in range(len(mutated_brain.bias_weights)):
        if np.random.random() < chance_to_mutate:
            mutated_brain.bias_weights[k] += np.random.uniform(mutation_range[0], mutation_range[1])
    return mutated_brain  # Возвращаем новый экземпляр



def mutate_n_weights(brain: BrainData, num_weights_to_mutate: int, mutation_range: Tuple[float, float]) -> BrainData:
    # Изменить N весов
    mutated_brain = deepcopy(brain)

    # Flatten weights and biases into a single list of references to mutate
    all_weights = [(layer, idx) for layer in mutated_brain.weights for idx in range(len(layer))]
    all_biases = [(mutated_brain.bias_weights, idx) for idx in range(len(mutated_brain.bias_weights))]
    all_parameters = all_weights + all_biases

    # Randomly select indices to mutate
    indices_to_mutate = np.random.choice(len(all_parameters), size=num_weights_to_mutate, replace=False)

    for idx in indices_to_mutate:
        param_list, param_idx = all_parameters[idx]
        param_list[param_idx] += np.random.uniform(mutation_range[0], mutation_range[1])

    return mutated_brain

"""
if self.mutation_algorithm == 0:
    # Выбрать лучший мозг и на его основе создать новых существ
    best_creature = deepcopy(self.get_sorted_creatures()[0])
    best_brain = deepcopy(best_creature.brains_data)
    best_brain_bias_layers = deepcopy(best_creature.bias_layers)
    self.kill_all_creatures()
    for i in range(self.creatures_per_generation - 1 - self.new_random_creatures):
        new_brain = deepcopy(best_brain)
        new_bias_layers = deepcopy(best_brain_bias_layers)
        edited = 0
        for j in range(len(new_brain)):
            for k in range(len(new_brain[j])):
                if np.random.random() < self.chance_to_mutate:
                    new_brain[j][k] += np.random.uniform(self.mutation_range[0],
                                                         self.mutation_range[1])
                    edited += 1
            if np.random.random() < self.chance_to_mutate / 2:
                new_bias_layers[j] += np.random.uniform(self.mutation_range[0],
                                                        self.mutation_range[1])
                edited += 1
        self.add_creature(new_brain, new_bias_layers)
    self.add_creature(best_brain, best_brain_bias_layers)

elif self.mutation_algorithm == 1:
    # Выбрать лучший мозг и на его основе создать новых существ
    best_creatures = deepcopy(self.get_sorted_creatures())
    best_creature = deepcopy(best_creatures[np.random.randint(0, 1)])
    best_brain = deepcopy(best_creature.brains_data)
    best_brain_bias_layers = deepcopy(best_creature.bias_layers)
    self.kill_all_creatures()

    for i in range(self.creatures_per_generation - 1 - self.new_random_creatures):
        new_brain = deepcopy(best_brain)
        new_bias_layers = deepcopy(best_brain_bias_layers)
        weights_to_mutate = np.random.randint(0, self.max_amount_of_mutated_weights)
        for j in range(weights_to_mutate):
            # Шанс на то, что будет изменён обычный вес. Иначе будет изменен вес смещения
            if np.random.random() < 0.9:
                rand_layer = np.random.randint(0, len(new_brain))
                new_brain[rand_layer][np.random.randint(0, len(new_brain[rand_layer]))] += \
                    np.random.uniform(self.mutation_range[0], self.mutation_range[1])
            else:
                new_bias_layers[np.random.randint(0, len(new_bias_layers))] += \
                    np.random.uniform(self.mutation_range[0], self.mutation_range[1])
        self.add_creature(new_brain, new_bias_layers)
    self.add_creature(best_brain, best_brain_bias_layers)

elif self.mutation_algorithm == 2:
    best_creatures = deepcopy(self.get_sorted_creatures()[:self.amount_of_best_creatures])
    # np.random.shuffle(best_creatures)
    self.kill_all_creatures()

    best_brains = []
    best_brains_bias_layers = []
    for c in deepcopy(best_creatures):
        best_brains.append(deepcopy(c.brains_data))
        best_brains_bias_layers.append(deepcopy(c.bias_layers))

    for i in range(self.creatures_per_generation - self.amount_of_best_creatures - 1 -
                   self.new_random_creatures):
        new_brain = deepcopy(best_brains[0])
        new_bias_layers = deepcopy(best_brains_bias_layers[0])
        second_index = np.random.randint(1, len(best_brains))
        second_brain = deepcopy(best_brains[second_index])
        second_bias_layers = deepcopy(best_brains_bias_layers[second_index])

        percent = np.random.randint(0, 50)
        replace_random_weights(new_brain, second_brain, percent=percent)
        replace_random_bias(new_bias_layers, second_bias_layers, percent=percent)

        new_brain = modify_brain(
            List(new_brain),
            np.random.randint(
                self.max_amount_of_mutated_weights // 2,
                self.max_amount_of_mutated_weights
            ),
            self.mutation_range
        )

        change_mask = np.random.randint(0, 2, len(new_bias_layers))
        new_bias_layers[change_mask] += np.random.uniform(self.mutation_range[0],
                                                          self.mutation_range[1])

        self.add_creature(new_brain, new_bias_layers)

    # Add the best creatures from previous generation
    for i in range(self.amount_of_best_creatures):
        self.add_creature(best_brains[i], best_brains_bias_layers[i])
"""
