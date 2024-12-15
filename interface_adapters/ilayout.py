from abc import abstractmethod, ABC
from typing import List


class IguiElement(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def remove(self):
        pass

    @abstractmethod
    def draw(self):
        pass


class ILayout(ABC):
    @abstractmethod
    def __init__(self,
                 gui_elements: List[IguiElement]):
        self.gui_elements = gui_elements

    @abstractmethod
    def draw(self):
        for element in self.gui_elements:
            element.draw()

    @abstractmethod
    def remove(self):
        for element in self.gui_elements:
            element.remove()