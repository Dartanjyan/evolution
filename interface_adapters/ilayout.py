from abc import ABC, abstractmethod
from typing import Tuple, Callable


class IWidget(ABC):
    """
    Базовый интерфейс для всех виджетов.
    """

    @abstractmethod
    def __init__(
        self,
        widget_id: str,
        position: Tuple[float, float],
        pos_anchor: Tuple[str, str]
    ) -> None:
        """
        Инициализация виджета.

        :param widget_id: Уникальный идентификатор виджета.
        :param position: Позиция элемента (x, y).
        :param pos_anchor: Точка привязки позиции (например, "center").
        """
        pass


class ITextWidget(IWidget, ABC):
    """
    Интерфейс для виджетов, которые содержат текст.
    """

    @abstractmethod
    def __init__(
        self,
        text: str,
        text_size: float,
        text_align: Tuple[str, str]
    ) -> None:
        """
        Инициализация текстового виджета.

        :param text: Текст виджета.
        :param text_size: Размер текста.
        :param text_align: Выравнивание текста (по горизонтали и вертикали).
        """
        pass


class IButton(ITextWidget, ABC):
    """
    Интерфейс для кнопок.
    """

    @abstractmethod
    def __init__(self, handler: Callable[[], None]) -> None:
        """
        Инициализация кнопки.

        :param handler: Функция-обработчик события (например, нажатия кнопки).
        """
        pass
