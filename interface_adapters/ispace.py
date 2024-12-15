from abc import ABC, abstractmethod


class ISpace(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def add(self, *args):
        pass

    @abstractmethod
    def remove(self, *args):
        pass

    @abstractmethod
    def step(self, delta: float):
        pass
