from typing import List, Optional, Tuple, Union

import numpy as np

from entities.creature.brain import BrainData
from interface_adapters.ibodyparts import IBodyPart, IJoint
from interface_adapters.ispace import ISpace


class Creature:
    def __init__(self,
                 body_parts: List[IBodyPart],
                 joints: List[IJoint],
                 brains: BrainData,
                 root_part: Optional[IBodyPart] = None,
                 sight_part: Optional[IBodyPart] = None,
                 memory_number: Optional[float | np.float32] = None,
                 immunity_gens: int = 0,
                 space: Optional[ISpace] = None):
        self.space: Optional[ISpace] = None
        self.body_parts = body_parts
        self.joints = joints
        self.root_part = root_part or self.body_parts[0]
        self.root_part.is_root_part = True
        self.sight_part = sight_part or self.root_part
        self.sight_part.is_sight_part = True
        self.brains = brains
        self.memory_number = memory_number
        self.immunity_gens = immunity_gens

        if space:
            self.add_to_space(space)

    def add_to_space(self, space: ISpace):
        if self.space:
            print("WARNING: Adding not killed creature to space")
        self.space = space
        self.space.add(self.body_parts, self.joints)

    def kill(self):
        if not self.space:
            raise ValueError("Can't kill creature before added to space")
        else:
            self.space.remove(self.body_parts, self.joints)
            self.space = None

    def move_to(self, pos: Union[List[float, float], Tuple[float, float]]):
        for part in self.body_parts:
            part.move_to(pos)

    def get_position(self) -> Tuple[float, float]:
        return self.root_part.get_position()
