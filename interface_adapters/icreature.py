from __future__ import annotations  # Включает поддержку отложенной аннотации

from abc import abstractmethod
from typing import List, Optional


class IBodyPart:
    @abstractmethod
    def __init__(self,
                 part_id: int,
                 vertices: List[float],
                 density: float,
                 friction: float,
                 elasticity: float,
                 is_root_part: bool = False,
                 is_sight_part: bool = False,
                 connect_to: Optional[IBodyPart] = None):
        self.part_id = part_id
        self.vertices = vertices
        self.density = density
        self.friction = friction
        self.elasticity = elasticity
        self.is_root_part = is_root_part
        self.is_sight_part = is_sight_part
        self.connect_to = connect_to

        self.is_part_body_original = connect_to is None


