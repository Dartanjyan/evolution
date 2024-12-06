import numpy as np
import pymunk  # as pymunk
from numba.typed import List
from pymunk import Vec2d as vec2d
from frameworks_drivers.config import *


class PolySegment:
    """Creates a Poly object\n
    If given another_body is None then will be created new Body"""

    def __init__(self,
                 segment_id: int,
                 another_body: pymunk.Body | None,
                 vertices,
                 density: float = POLY_DENSITY,
                 friction: float = POLY_FRICTION,
                 elasticity: float = 0.2,
                 is_root_part: bool = False,
                 is_sight_part: bool = False
                 ):
        self.is_sight_part = is_sight_part
        self.is_root_part = is_root_part
        self.id = segment_id
        self.is_body_original = another_body is None

        self.vertices = vertices

        if self.is_body_original:
            self.body = pymunk.Body()
            self.body.position = (0, 0)
        else:
            self.body = another_body

        t = pymunk.Vec2d(0, 0)
        s = 0
        for i in vertices:
            t += i
            s += 1
        t /= s
        is_body_new = another_body is None
        self.body = pymunk.Body() if another_body is None else another_body
        self.shape = pymunk.Poly(
            self.body,
            vertices,
            pymunk.Transform(tx=0, ty=0)
        )
        self.shape.density = density
        self.shape.friction = friction
        self.shape.elasticity = elasticity
        self.shape.filter = pymunk.ShapeFilter(1)


class Joint:
    def __init__(self,
                 joint_id: int,
                 body_a: pymunk.Body,
                 body_b: pymunk.Body,
                 anchor_a: tuple | vec2d,
                 anchor_b: tuple | vec2d = None,
                 rest_angle: float = 0,
                 stiffness: float = 8e5,
                 damping: float = 4e4,
                 collide_bodies: bool = False
                 ):
        self.id = joint_id

        if anchor_b is None:
            anchor_b = anchor_a

        self.pin_joint = pymunk.PinJoint(body_a, body_b, anchor_a, anchor_b)
        self.pin_joint.collide_bodies = collide_bodies
        self.pin_joint.error_bias = pow(1.0 - 0.01, 1200000)

        self.damped_joint = pymunk.DampedRotarySpring(body_a, body_b, rest_angle, stiffness, damping)
        self.damped_joint.collide_bodies = collide_bodies
        self.damped_joint.max_force = 0.1


class Bone:
    def __init__(self,
                 bone_id: int,
                 pos1: pymunk.Vec2d,
                 pos2: pymunk.Vec2d,
                 r: float = BONE_SHAPE_WIDTH,
                 density: float = BONE_DENSITY,
                 friction: float = 0.9,
                 is_root_part: bool = False,
                 is_sight_part: bool = False
                 ):
        self.is_sight_part = is_sight_part
        self.is_root_part = is_root_part
        self.id = bone_id
        self.pos1 = pos1
        self.pos2 = pos2

        self.body = pymunk.Body()
        self.body.position = (0, 0)
        # Shape setup
        self.shape = pymunk.Segment(self.body, pos1, pos2, r)
        self.shape.density = density
        self.shape.friction = friction
        self.shape.filter = pymunk.ShapeFilter(1)


class Creature:
    def __init__(self,
                 polies: list[PolySegment],
                 bones: list[Bone],
                 joints: list[Joint],
                 root_part: PolySegment | Bone,
                 sight_part: PolySegment | Bone = None,
                 brains_data: List[np.ndarray] = None,
                 bias_layers: List = None,
                 memory_number: float | np.float32 = None,
                 immunity_gens: int = 0
                 ):
        if polies is None:
            polies = []
        if bones is None:
            bones = []
        if joints is None:
            joints = []

        if memory_number is None:
            self.memory_number = 0

        # TODO потом сделать так чтобы одно существо могло жить несколько поколений подряд
        self.immunity_gens: int = immunity_gens

        self.space = None

        self.polies: list[PolySegment] = polies
        self.bones: list[Bone] = bones
        self.joints: list[Joint] = joints
        self.root_part: PolySegment | Bone = root_part
        if sight_part is None:
            self.sight_part = self.root_part
        else:
            self.sight_part = sight_part
        self.brains_data: List[np.ndarray] = brains_data
        self.bias_layers: List = bias_layers

        self.root_part.is_root_part = True
        self.sight_part.is_sight_part = True

    def add_to_space(self,
                     space: pymunk.Space):
        self.space = space
        for poly in self.polies:
            if poly.is_body_original:
                self.space.add(poly.body)
            self.space.add(poly.shape)
        for bone in self.bones:
            self.space.add(bone.body, bone.shape)
        for joint in self.joints:
            self.space.add(joint.pin_joint, joint.damped_joint)

    def kill(self):
        for poly in self.polies:
            if poly.is_body_original:
                self.space.remove(poly.body)
            self.space.remove(poly.shape)
        for bone in self.bones:
            self.space.remove(bone.body, bone.shape)
        for joint in self.joints:
            self.space.remove(joint.pin_joint, joint.damped_joint)

    def move_to(self, pos: pymunk.Vec2d | tuple[float, float]):
        for part in self.bones + self.polies:
            part.body.position = pos

    def get_position(self):
        point = np.mean(np.array([[v.rotated(self.root_part.body.angle)[0] + self.root_part.body.position[0],
                                   v.rotated(self.root_part.body.angle)[1] + self.root_part.body.position[1]]
                                  for v in self.root_part.shape.get_vertices()]), axis=0)
        return pymunk.Vec2d(point[0], point[1])
