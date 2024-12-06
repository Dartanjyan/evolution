WIDTH: float = 800
HEIGHT: float = 600
TITLE = "Physics simulation"
FPS = 60

STEP_DIVIDER = 5
PHYSICS_STEP = 0.01 / STEP_DIVIDER
PHYSICS_IPS = 120  # Iterations per second

NN_CALC_STEP = 60//6  # Количество кадров между каждой обработкой нейросети. 60//вызовы-в-секунду
GENERATION_TIME = 10*FPS  # 10 сек

GRAVITY = 0, 981

GRAY = (120, 120, 120)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PINK = (230, 50, 230)
CYAN = (0, 255, 255)

BONE_COLOR = (115, 126, 137)
BONE_DRAW_WIDTH = 5
BONE_SHAPE_WIDTH = BONE_DRAW_WIDTH/2
BONE_DENSITY = 0.01 / 2
BONE_FRICTION = 1

FLOOR_FRICTION = 1

POLY_COLOR = (170, 153, 137)
POLY_DENSITY = BONE_DENSITY * 2 * 2
POLY_FRICTION = BONE_FRICTION

MUSCLE_COLOR = (255, 129, 110)
MUSCLE_WIDTH = 3

FLOOR_COLOR = (79, 73, 85)


class activation:
    cos = -2
    sin = -1
    tanh = 0
    elu = 1
    sigmoid = 2
    relu = 3
    softmax = 4
    root = 5
    nothing = 6
    line_gate = 7
    half_sin = 8

