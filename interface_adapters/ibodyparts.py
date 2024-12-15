from __future__ import annotations  # Включает поддержку отложенной аннотации

from abc import abstractmethod, ABC
from typing import List, Optional, Tuple, Union


class IBodyPart(ABC):
    @abstractmethod
    def __init__(self,
                 part_id: int,
                 vertices: List[float],
                 density: float,
                 friction: float,
                 elasticity: float,
                 is_root_part: bool = False,
                 is_sight_part: bool = False,
                 connect_to: Optional[IBodyPart] = None) -> None:
        self.part_id = part_id
        self.vertices = vertices
        self.density = density
        self.friction = friction
        self.elasticity = elasticity
        self.is_root_part = is_root_part
        self.is_sight_part = is_sight_part
        self.connect_to = connect_to
        self.is_part_body_original = connect_to is None

    @abstractmethod
    def get_position(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def move_to(self, pos: Union[List[float, float], Tuple[float, float]]) -> None:
        pass

    @abstractmethod
    def get_angle(self) -> float:
        pass

class IJoint(ABC):
    @abstractmethod
    def __init__(self,
                 joint_id: int,
                 part_a: IBodyPart,
                 part_b: IBodyPart,
                 anchor_a: Tuple[float, float],
                 anchor_b: Optional[Tuple[float, float]],
                 stiffness: float,
                 damping: float,
                 rest: float = 0.0,
                 collide_bodies: bool = False) -> None:
        self.joint_id = joint_id
        self.part_a = part_a
        self.part_b = part_b
        self.anchor_a = anchor_a
        if anchor_b is None:
            anchor_b = anchor_a
        self.anchor_b = anchor_b
        self.stiffness = stiffness
        self.damping = damping
        self.rest = rest
        self.collide_bodies = collide_bodies

    @abstractmethod
    def set_rest(self) -> None:
        pass
