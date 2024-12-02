import os
import pickle
import platform
import random
import sys
import threading
import time
from copy import deepcopy
from sys import argv
import math

import matplotlib.pyplot as plt
import numpy as np
import pygame
import pygame.locals
import pymunk
import pymunk.pygame_util
from numba import njit
from numba.typed import List
from scipy.interpolate import UnivariateSpline

import myCreature
from settings import *

np.set_printoptions(precision=8, suppress=True, formatter={'all': lambda x: f'{x:0.2f}'}, linewidth=100)

seed = random.randrange(2 ** 32)
# seed = 1
print("Seed:", seed)
np.random.seed(seed)
start = time.time()

if "--mut-alg" in argv:
    mut_alg = int(argv[argv.index("--mut-alg")+1])
else:
    mut_alg = 2

user_name = os.getlogin()

if platform.system() == 'Windows':
    save_dir = f'C:\\Users\\{user_name}\\evolutionByAlex\\saves'
elif platform.system() == 'Linux':
    save_dir = f'/home/{user_name}/evolutionByAlex/saves'
else:
    save_dir = "saves"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created saves folder {save_dir}")
save_name = f"autosave{mut_alg}.save"
save_path = os.path.join(save_dir, save_name)

def smooth_curve_spline(points, smoothing_factor):
    x = np.arange(len(points))
    spline = UnivariateSpline(x, points, s=smoothing_factor)
    return spline(x)


@njit
def calc_next_layer(i, h, b, function: int):
    x = np.dot(i, h) * b
    if function == -2:
        return np.cos(x)  # cos
    elif function == -1:
        return np.sin(x)  # sin
    elif function == 0:
        return np.tanh(x)  # tanh
    elif function == 1:
        alpha = 1.0
        y = np.where(x > 0, x, alpha * (np.exp(x) - 1))  # elu
        return y
    elif function == 2:
        return (1 / (1 + np.e ** (-x))) * 2 - 1  # sigmoid
    elif function == 3:
        x[x < 0] = 0
        return x  # relu
    elif function == 4:
        e_x = np.exp(x)
        sum_e_x = np.sum(e_x)
        return e_x / sum_e_x  # softmax
    elif function == 5:
        y = np.sqrt(np.abs(x))
        y[x < 0] *= -1
        return y  # root
    elif function == 6:
        return x  # no activation
    elif function == 7:
        x /= np.pi
        x[x < -1] = -1
        x[x > 1] = 1
        return x  # simple line_gate gate
    elif function == 8:
        y = np.sin(x)
        y[x > np.pi] = np.pi
        y[x < -np.pi] = -np.pi
        return y  # cut sin


@njit
def calc_nn(brains_data: List[np.ndarray],
            bias_layers: List,
            __current_data,
            hidden_activation_func: int,
            out_activation_func: int):
    for i in range(len(brains_data)):
        # Separate last layer from others to use different activation functions
        if i < len(brains_data) - 1:
            __current_data = calc_next_layer(__current_data, brains_data[i], bias_layers[i],
                                             hidden_activation_func)
        else:
            __current_data = calc_next_layer(__current_data, brains_data[i], bias_layers[i],
                                             out_activation_func)
    return __current_data


def replace_random_weights(list1, list2, percent):
    # Проверка на соответствие размерности массивов
    if len(list1) != len(list2):
        print("Списки должны содержать одинаковое количество массивов")
        return

    # Проход по каждому массиву в списках
    for i in range(len(list1)):
        array1 = list1[i]
        array2 = list2[i]

        # Получение размерности текущего массива
        shape = array1.shape

        # Подсчет количества элементов для замены
        total_elements = np.prod(shape)
        num_elements_to_replace = int(percent / 100 * total_elements)

        # Получение случайных индексов для замены
        random_indices = np.random.choice(total_elements, num_elements_to_replace, replace=False)

        # Преобразование индексов в многомерные индексы
        indices = np.unravel_index(random_indices, shape)

        # Замена значений из первого списка значениями из второго списка по соответствующим индексам
        array1[indices] = array2[indices]


@njit
def replace_random_bias(list1, list2, percent):
    # Проверка на соответствие размерности списков
    if len(list1) != len(list2):
        print("Списки должны иметь одинаковую длину")
        return

    # Расчет количества элементов для замены
    num_elements = len(list1)
    num_elements_to_replace = int(percent / 100 * num_elements)

    # Получение случайных индексов для замены
    random_indices = np.random.choice(num_elements, num_elements_to_replace, replace=False)

    # Замена значений из первого списка значениями из второго списка по соответствующим индексам
    for index in random_indices:
        list1[index] = list2[index]


@njit
def modify_brain(new_brain, num_changes, mut_range):
    """
Изменяет заданное количество чисел из списка new_brain на число из диапазона (-1; 1)

Args:
new_brain: список, который содержит двумерные массивы чисел numpy разных размерностей
num_changes: количество чисел, которые необходимо изменить

Returns:
Модифицированный список
"""
    for b in range(len(new_brain)):
        for _ in range(num_changes):
            i = np.random.randint(0, new_brain[b].shape[0])
            j = np.random.randint(0, new_brain[b].shape[1])
            add = np.random.uniform(mut_range[0], mut_range[1])
            new_brain[b][i, j] += add
    return list(new_brain)


class App:
    def __init__(self):
        self.generation = 0
        self.creatures_per_generation = 50
        self.new_random_creatures = 3

        self.chance_to_mutate = 10 / 100  # для алгоритма 0
        self.max_amount_of_mutated_weights = 3  # для алгоритмов 1, 2
        self.mutation_range = -3, 3  # для алгоритмов 0, 1, 2
        self.random_brains_uniform = -3, 3
        self.mutation_algorithm = mut_alg
        self.amount_of_best_creatures = 4  # для алгоритма 2

        self.mutation_names = {
            0: f"Select 1 creature by best x; mutate every weight with chance {self.chance_to_mutate * 100}%\nin range "
               f"{self.mutation_range}",
            1: f"Select 1 creature by best x; mutate defined amount of weights with "
               f"limit of {self.max_amount_of_mutated_weights}",
            2: f"Select {self.amount_of_best_creatures} creatures by best x; combine their weights;\nmutate defined amount of weights with "
               f"limit of {self.max_amount_of_mutated_weights}"
        }

        self.timer_for_fps_update = 0
        self.debug_draw = False
        self.draw = False

        self.space = pymunk.Space()
        self.space.gravity = GRAVITY

        pygame.init()
        self.screen = pygame.display.set_mode([WIDTH, HEIGHT])

        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN])
        pygame.display.set_caption(f"Evolution by Alex ({mut_alg = })")
        self.fps_clock = pygame.time.Clock()
        self.physics_clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.running = True

        floor_len = 2 ** 31 - 1
        self.floor_rect = pygame.Rect([0, HEIGHT // 3 * 2], [WIDTH, HEIGHT // 3])
        self.floor = pymunk.Poly(self.space.static_body, [(-floor_len / 2, HEIGHT // 3 * 2),
                                                          (floor_len / 2, HEIGHT // 3 * 2),
                                                          (floor_len / 2, HEIGHT),
                                                          (-floor_len / 2, HEIGHT)])
        self.floor.friction = FLOOR_FRICTION
        self.floor.elasticity = 0
        self.floor.filter = pymunk.ShapeFilter(0)
        self.space.add(self.floor)

        self.my_creatures: list[myCreature.Creature] = []
        self.hidden_layers = [13]
        self.best_x = [0]

        self.physics_thread = threading.Thread(target=self.run_simulation_physics)
        self.calc_all_creatures_thread = None
        self.ai_calculating = False
        self.threads = os.cpu_count()
        self.creatures_per_thread = np.ceil(self.creatures_per_generation / self.threads)

        self.font = pygame.font.Font(None, 21)
        self.lines = """"D" to enable/disable rendering
"P" to switch rendering between pygame and pymunk debug draw
"L" - Load autosave
Generation: 0
FPS: """.split('\n')
        self.lines_of_text_to_blit = [self.font.render(line, True, BLACK) for line in self.lines]

    def calculate_creature_ai(self, creature: myCreature.Creature):
        save = False
        # На вход подаются:
        #   - Просто число из прошлого шага нейросети
        #   - Угол поворота root_part
        #   - Угол поворота каждой части тела относительно угла поворота root_part

        # На выходе:
        #   - просто число для следующей итерации
        #   - rest_angle для каждого joint

        current_data = np.array([creature.memory_number, creature.root_part.body.angle / (2 * np.pi)])
        for part in creature.polies + creature.bones:
            if not part.is_root_part:
                # Рассчитать угол поворота части тела относительно коренной, в пределе [0; 1]
                normalized_angle = ((creature.root_part.body.angle - part.body.angle) % (2 * np.pi)) / (2 * np.pi)
                current_data = np.append(current_data, normalized_angle)

        h_a, o_a = activation.sigmoid, activation.tanh

        if save:
            f = open("out.txt", "w")
            f.write("Input data:\n" + str(current_data))
            __current_data = deepcopy(current_data)

        # s = time.perf_counter_ns()
        current_data = calc_nn(List(creature.brains_data), List(creature.bias_layers), current_data,
                               h_a,
                               o_a)
        # print(round((time.perf_counter_ns() - s) / 1e6, 3), "ms")

        if save:
            f.write("\n\nOutput data:\n" + str(current_data))
            f.write("\n\n" + "-" * 30 + "\n\nCalculations:\n")
            brains_data = creature.brains_data
            bias_layers = creature.bias_layers

            for i in range(len(brains_data)):
                f.write(f"Iteration {i}: {__current_data} -> ")
                # Separate last layer from others to use different activation functions
                if i < len(brains_data) - 1:
                    __current_data = calc_next_layer(__current_data, brains_data[i], bias_layers[i],
                                                     h_a)
                else:
                    __current_data = calc_next_layer(__current_data, brains_data[i], bias_layers[i],
                                                     o_a)
                f.write(str(__current_data) + "\n")
                """
            f.write("\n\n" + "-" * 30 + "\n\nWeights:\n")
            for i in range(len(creature.brains_data)):
                f.write(f"Layer {i}: {creature.brains_data[i]} {creature.bias_layers[i]}\n")
                """
            f.close()

        # Applying data to creature
        creature.memory_number = current_data[0]
        for i in range(1, len(creature.joints)):
            creature.joints[i].damped_joint.rest_angle = np.pi * 0.3 * (current_data[i] * 1 - 0)

    def do_step(self):
        for i in range(int(STEP_DIVIDER)):
            self.space.step(PHYSICS_STEP)

    def add_creature(self, brains_data=None, bias_layers=None):
        creature_id = 2

        polies = []
        bones = []
        joints = []

        bias_coordinates = (0, HEIGHT / 2)

        if creature_id == 1:
            poly_body = myCreature.PolySegment(
                self.space, 1, None,
                (
                    pymunk.Vec2d(150, 105),
                    pymunk.Vec2d(250, 105),
                    pymunk.Vec2d(250, 125),
                    pymunk.Vec2d(150, 125)
                )
            )

            lleg1 = myCreature.Bone(self.space, 1, pymunk.Vec2d(150, 120),
                                    pymunk.Vec2d(130, 140))
            lleg2 = myCreature.Bone(self.space, 2, pymunk.Vec2d(130, 140),
                                    pymunk.Vec2d(150, 160))
            rleg1 = myCreature.Bone(self.space, 3, pymunk.Vec2d(250, 120),
                                    pymunk.Vec2d(230, 140))
            rleg2 = myCreature.Bone(self.space, 4, pymunk.Vec2d(230, 140),
                                    pymunk.Vec2d(250, 160))

            ljoint1 = myCreature.Joint(self.space, 1, poly_body.body, lleg1.body,
                                       pymunk.Vec2d(150, 120),
                                       pymunk.Vec2d(150, 120))
            ljoint2 = myCreature.Joint(self.space, 2, lleg1.body, lleg2.body, pymunk.Vec2d(130, 140),
                                       pymunk.Vec2d(130, 140))
            rjoint1 = myCreature.Joint(self.space, 3, poly_body.body, rleg1.body,
                                       pymunk.Vec2d(250, 120),
                                       pymunk.Vec2d(250, 120))
            rjoint2 = myCreature.Joint(self.space, 4, rleg1.body, rleg2.body, pymunk.Vec2d(230, 140),
                                       pymunk.Vec2d(230, 140))

            polies.extend([poly_body])
            bones.extend([lleg1, lleg2, rleg1, rleg2])
            joints.extend([ljoint1, ljoint2, rjoint1, rjoint2])

        elif creature_id == 2:
            p1 = myCreature.PolySegment(
                self.space, 1, None,
                (
                    pymunk.Vec2d(40, 20),
                    pymunk.Vec2d(140, 20),
                    pymunk.Vec2d(120, 50)
                )
            )

            ll1 = myCreature.Bone(self.space, 1, pymunk.Vec2d(40, 20),
                                  pymunk.Vec2d(80, 40))
            ll2 = myCreature.Bone(self.space, 2, pymunk.Vec2d(80, 40),
                                  pymunk.Vec2d(40, 60))
            ll3 = myCreature.Bone(self.space, 3, pymunk.Vec2d(40, 60),
                                  pymunk.Vec2d(60, 80))
            rl1 = myCreature.Bone(self.space, 4, pymunk.Vec2d(140, 20),
                                  pymunk.Vec2d(160, 40))
            rl2 = myCreature.Bone(self.space, 5, pymunk.Vec2d(160, 40),
                                  pymunk.Vec2d(140, 60))
            rl3 = myCreature.Bone(self.space, 6, pymunk.Vec2d(140, 60),
                                  pymunk.Vec2d(160, 60))
            t1 = myCreature.Bone(self.space, 7, pymunk.Vec2d(40, 20),
                                 pymunk.Vec2d(0, 0))
            h1 = myCreature.Bone(self.space, 8, pymunk.Vec2d(140, 20),
                                 pymunk.Vec2d(160, 0))

            stiffness = 8e5
            damping = 4e4
            j1 = myCreature.Joint(self.space, 1, p1.body, ll1.body, pymunk.Vec2d(40, 20),
                                  pymunk.Vec2d(40, 20), stiffness=stiffness, damping=damping)
            j2 = myCreature.Joint(self.space, 2, ll1.body, ll2.body, pymunk.Vec2d(80, 40),
                                  pymunk.Vec2d(80, 40), stiffness=stiffness, damping=damping)
            j3 = myCreature.Joint(self.space, 3, ll2.body, ll3.body, pymunk.Vec2d(40, 60),
                                  pymunk.Vec2d(40, 60), stiffness=stiffness, damping=damping)
            j4 = myCreature.Joint(self.space, 4, p1.body, rl1.body, pymunk.Vec2d(140, 20),
                                  pymunk.Vec2d(140, 20), stiffness=stiffness, damping=damping)
            j5 = myCreature.Joint(self.space, 5, rl1.body, rl2.body, pymunk.Vec2d(160, 40),
                                  pymunk.Vec2d(160, 40), stiffness=stiffness, damping=damping)
            j6 = myCreature.Joint(self.space, 6, rl2.body, rl3.body, pymunk.Vec2d(140, 60),
                                  pymunk.Vec2d(140, 60), stiffness=stiffness, damping=damping)
            j7 = myCreature.Joint(self.space, 7, p1.body, t1.body, pymunk.Vec2d(40, 20),
                                  pymunk.Vec2d(40, 20), stiffness=stiffness, damping=damping)
            j8 = myCreature.Joint(self.space, 8, p1.body, h1.body, pymunk.Vec2d(140, 20),
                                  pymunk.Vec2d(140, 20), stiffness=stiffness, damping=damping)

            polies.extend([p1])
            bones.extend([ll1, ll2, ll3, rl1, rl2, rl3, t1, h1])
            joints.extend([j1, j2, j3, j4, j5, j6, j7, j8])

        if brains_data is None:
            brains_data = []

            # memory_number + угол root_part + угол поворота каждой кости и поли относительно root_part
            input_len = 1 + len(polies) + len(bones)
            # memory_number + rest_len для каждого joint
            output_len = 1 + len(joints)

            a, b = self.random_brains_uniform
            # Случайная генерация весов
            brains_data.append(np.random.uniform(a, b, size=(input_len, self.hidden_layers[0])))
            for i in range(max(len(self.hidden_layers) - 1, 0)):
                brains_data.append(
                    np.random.uniform(a, b, size=(self.hidden_layers[i], self.hidden_layers[i + 1]))
                )
            brains_data.append(np.random.uniform(a, b, size=(self.hidden_layers[-1], output_len)))

            for i in range(len(brains_data)):
                brains_data[i] = np.asarray(brains_data[i])
            # print(brains_data)
        if bias_layers is None:
            a, b = self.random_brains_uniform
            bias_layers = np.random.uniform(a, b, size=len(brains_data))
            # print(bias_layers)

        creature = myCreature.Creature(polies=polies, bones=bones, joints=joints, root_part=polies[0],
                                       brains_data=brains_data, bias_layers=bias_layers)
        creature.move_to(bias_coordinates)
        self.my_creatures.append(creature)

    def get_sorted_creatures(self):
        creatures = sorted(self.my_creatures, key=lambda creature: creature.get_position().x)[::-1]
        return creatures

    def run_simulation_physics(self):
        self.do_step()
        if self.draw:
            self.physics_clock.tick(PHYSICS_IPS)

    def kill_all_creatures(self):
        for c in self.my_creatures:
            for p in c.polies:
                self.space.remove(p.body, p.shape)
            for b in c.bones:
                self.space.remove(b.body, b.shape)
            for j in c.joints:
                self.space.remove(j.damped_joint, j.pin_joint)
        self.my_creatures: list[myCreature.Creature] = []

    def save(self):
        save_data = {
            "generation": self.generation,
            "current_brains": [{"brain": i.brains_data, "bias": i.bias_layers} for i in self.my_creatures],
            "best_x": self.best_x,
            "hidden_layers_config": self.hidden_layers
        }
        with open(save_path, "wb") as f:
            pickle.dump(save_data, f)
        print(f"Saved automatically. Generation: {self.generation}. Best x: {max(self.best_x)}")

    def load(self):
        if os.path.exists(save_path):
            if self.physics_thread.is_alive():
                self.physics_thread.join()
            self.kill_all_creatures()
            with open(save_path, "rb") as f:
                load_data = pickle.load(f)
            self.generation = load_data["generation"]
            self.best_x = load_data["best_x"]
            for creature_load_data in load_data["current_brains"]:
                self.add_creature(creature_load_data["brain"], creature_load_data["bias"])
            self.hidden_layers = load_data["hidden_layers_config"]

            self.lines_of_text_to_blit[-2] = self.font.render(
                f"Generation: {self.generation}",
                True,
                pygame.Color("black")
            )

            layers_config, nodes = self.get_layers_and_nodes()
            print("\nLoaded save")
            print("Layers: ", layers_config, ", Nodes: ", nodes, ", Generation: ", self.generation, sep="")
        else:
            print(f"Save file {save_path} not found!")

    def get_layers_and_nodes(self):
        layers_config = []
        nodes = 0
        for i, matrix in enumerate(self.my_creatures[0].brains_data):
            layers_config.append(matrix.shape[0])
            nodes += matrix.size
            if i == len(self.my_creatures[0].brains_data) - 1:
                layers_config.append(matrix.shape[1])
        return layers_config, nodes

    def calc_all_creatures(self):
        for c in self.my_creatures:
            # s = time.perf_counter_ns()
            self.calculate_creature_ai(c)
            # print(round((time.perf_counter_ns() - s) / 1e6, 3), "ms")

    def blit_all_texts(self):
        # Blit all texts on screen
        for i in range(len(self.lines_of_text_to_blit)):
            self.screen.blit(self.lines_of_text_to_blit[i],
                             (5,
                              (self.lines_of_text_to_blit[
                                   min(0, i - 1)].get_bounding_rect().bottom + 10) * i + 10))

    def run(self):
        for i in range(self.creatures_per_generation):
            self.add_creature()

        counter_for_nn_calc = NN_CALC_STEP
        generation_counter = 0

        if "--load" in argv:
            self.load()

        # Count amount of the nodes
        layers_config, nodes = self.get_layers_and_nodes()
        print("Layers: ", layers_config, ", Nodes: ", nodes, sep="")
        print(f"Mutation algorithm: {self.mutation_algorithm} ({self.mutation_names[self.mutation_algorithm]})")

        if self.max_amount_of_mutated_weights is None:
            self.max_amount_of_mutated_weights = nodes

        while self.running:
            self.physics_thread = threading.Thread(target=self.do_step)
            self.physics_thread.start()
            self.calc_all_creatures_thread = threading.Thread(target=self.calc_all_creatures)

            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.save()
                    self.running = False
                    self.kill_all_creatures()
                    self.physics_thread.join()
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_l:
                        """Кнопка L - Загрузить автосейв"""
                        self.load()
                        counter_for_nn_calc = NN_CALC_STEP
                        generation_counter = 0
                    elif event.key == pygame.K_SPACE:
                        """Пробел - удалить все существа"""
                        # self.kill_all_creatures()
                    elif event.key == pygame.K_p:
                        self.debug_draw = not self.debug_draw
                    elif event.key == pygame.K_d:
                        self.draw = not self.draw
                        self.screen.fill(WHITE)
                        pygame.display.flip()

            # Calculate every creature's NN
            counter_for_nn_calc += 1
            if counter_for_nn_calc >= NN_CALC_STEP:
                self.calc_all_creatures_thread.start()
                self.ai_calculating = True
                counter_for_nn_calc = 0

            # Create new generation
            generation_counter += 1
            if generation_counter >= GENERATION_TIME:
                generation_counter = 0
                self.generation += 1
                self.lines_of_text_to_blit[-2] = self.font.render(
                    f"Generation: {self.generation}",
                    True,
                    pygame.Color("black")
                )
                self.lines_of_text_to_blit[-1] = self.font.render(
                    "FPS: disabled",
                    True,
                    pygame.Color("black")
                )

                self.blit_all_texts()
                pygame.display.flip()

                __best_x = -float('inf')
                for i in self.my_creatures:
                    if (__b := i.get_position().x) > __best_x:
                        __best_x = __b
                self.best_x.append(__best_x)

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

                # Add some random creatures
                for i in range(self.new_random_creatures):
                    self.add_creature()

                # Autosave
                if self.generation % 100 == 0:
                    self.save()

                if self.generation == 1001 and False:
                    break

            # Rendering
            self.screen.fill(WHITE)
            if self.draw:
                # s = time.perf_counter_ns()
                if self.debug_draw:
                    self.space.debug_draw(self.draw_options)
                else:
                    pygame.draw.rect(self.screen, FLOOR_COLOR, self.floor_rect)
                    pygame.draw.line(self.screen, RED,
                                     (max(self.best_x), self.floor_rect.y),
                                     (max(self.best_x), self.floor_rect.y - 100),
                                     1)

                    for i, creature in enumerate(self.my_creatures):
                        polies = creature.polies
                        bones = creature.bones
                        old_id = self.creatures_per_generation - self.new_random_creatures - 1
                        for p in polies:
                            points = []
                            for v in p.shape.get_vertices():
                                x = v.rotated(p.body.angle)[0] + p.body.position[0]
                                y = v.rotated(p.body.angle)[1] + p.body.position[1]
                                points.append((x, y))
                            pygame.draw.polygon(self.screen, POLY_COLOR, points)
                            pygame.draw.aalines(self.screen, BLACK if i != old_id else RED, True, points)
                        for b in bones:
                            p1 = b.body.position + b.shape.a.rotated(b.body.angle)
                            p2 = b.body.position + b.shape.b.rotated(b.body.angle)

                            pygame.draw.line(self.screen, BLACK if i != old_id else RED, p1, p2, BONE_DRAW_WIDTH + 2)
                            pygame.draw.circle(self.screen, BLACK if i != old_id else RED, p1, BONE_DRAW_WIDTH / 1.8 + 1)
                            pygame.draw.circle(self.screen, BLACK if i != old_id else RED, p2, BONE_DRAW_WIDTH / 1.8 + 1)
                        for b in bones:
                            p1 = b.body.position + b.shape.a.rotated(b.body.angle)
                            p2 = b.body.position + b.shape.b.rotated(b.body.angle)

                            pygame.draw.line(self.screen, BONE_COLOR, p1, p2, BONE_DRAW_WIDTH)
                            pygame.draw.circle(self.screen, BONE_COLOR, p1, BONE_DRAW_WIDTH / 1.8)
                            pygame.draw.circle(self.screen, BONE_COLOR, p2, BONE_DRAW_WIDTH / 1.8)

                    select = False
                    if select:
                        best = self.get_sorted_creatures()[0]
                        pygame.draw.circle(self.screen, RED, best.get_position(), 10, 1)

                # print(int((time.perf_counter_ns() - s) / 1e6), "ms")
                # return 0
                self.fps_clock.tick(FPS)

            # Update fps text on screen
            if self.draw and time.time() - self.timer_for_fps_update > 1 / 5:
                self.timer_for_fps_update = time.time()
                _fps = str(round(self.fps_clock.get_fps(), 3)).ljust(6, '0')
                self.lines_of_text_to_blit[-1] = self.font.render(
                    f"FPS: {_fps} ({self.fps_clock.get_rawtime()}/{self.fps_clock.get_time()} ms)",
                    True,
                    pygame.Color("black")
                )

            # Blit all texts on screen
            self.blit_all_texts()

            if self.draw:
                pygame.display.flip()

            self.physics_thread.join()
            if self.ai_calculating:
                self.calc_all_creatures_thread.join()
                self.ai_calculating = False

        pygame.quit()

        if len(self.best_x) > 0:
            print(f"Max distance: {round(max(self.best_x), 3)} on generation {self.best_x.index(max(self.best_x))}")
        if len(self.best_x) > 100:
            """best_x_smooth = deepcopy(self.best_x)
            smooth_r = 10 ** (len(str(len(self.best_x)*2)) - 2)
            
            for i in range(len(best_x_smooth)):
                best_x_smooth[i] = np.mean(self.best_x[max(0, i - smooth_r):min(i + smooth_r + 1, len(best_x_smooth))])
            """
            if __name__ == "__main__":
                best_x_smooth = smooth_curve_spline(self.best_x,
                                                    len(self.best_x) * (80 * 2 ** (self.generation / (10 ** int(math.log(22315, 10))))))
                plt.plot(range(0, self.generation + 1), best_x_smooth)
                plt.suptitle(
                    f"Mutation algorithm: {self.mutation_algorithm}\n({self.mutation_names[self.mutation_algorithm]})")
                plt.ylabel("Distance")
                plt.xlabel("Generation")
                plt.grid()
                plt.show()
            else:
                print(f'Процесс "{__name__}" завершён (наверное)')
        return 1


if __name__ == "__main__":
    sys.exit(App().run())
