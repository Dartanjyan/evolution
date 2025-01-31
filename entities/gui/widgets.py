from typing import Union, Tuple, Optional


class Position:
    __slots__ = 'x', 'y', 'anchor_x', 'anchor_y'
    def __init__(self,
                 x: Union[str, int] = "50%",
                 y: Union[str, int] = "50%",
                 anchor_x: str = "center",
                 anchor_y: str = "center"
                 ) -> None:
        self.x = x
        self.y = y
        self.anchor_x = anchor_x
        self.anchor_y = anchor_y


class Background:
    __slots__ = 'color', 'img'
    def __init__(self,
                 color: Optional[Union[Tuple[int, int, int, int], Tuple[int, int, int], str]] = None,
                 img: Optional = None) -> None:
        self.color = color
        self.img = img


class Label:
    __slots__ = 'type', 'id', 'text', 'font_color', 'text_size', 'text_font', 'text_align_x', 'text_align_y', 'position'
    def __init__(self,
                 id: str,
                 text: str,
                 font_color: Union[Tuple[int, int, int, int], Tuple[int, int, int], str] = "black",
                 text_size: Union[int, str] = "20px",
                 text_font: str = "Arial",
                 text_align_x: str = "center",
                 text_align_y: str = "center",
                 position: Position = Position()
                 ) -> None:
        self.id = id
        self.text = text
        self.font_color = font_color
        self.text_size = text_size
        self.text_font = text_font
        self.text_align_x = text_align_x
        self.text_align_y = text_align_y
        self.position = position


class PressButton:
    __slots__ = ('id', 'text', 'x', 'y',
                 'text_align_x', 'text_align_y', 'width', 'height', 'position',
                 'text_size', 'hover_text_size', 'pressed_text_size',
                 'text_font', 'hover_text_font', 'pressed_text_font',
                 'hover_text', 'pressed_text',
                 'font_color', 'hover_font_color', 'pressed_font_color',
                 'background_color', 'hover_background_color', 'pressed_background_color',
                 'border_width', 'hover_border_width', 'pressed_border_width',
                 'border_color', 'hover_border_color', 'pressed_border_color')

    def __init__(self,
                 id: str,
                 text: str,
                 x: int,
                 y: int,
                 width: Union[str, int],
                 height: Union[str, int],
                 hover_text: Optional[str] = None,
                 pressed_text: Optional[str] = None,
                 text_align_x: str = "center",
                 text_align_y: str = "center",
                 position: Position = Position(),
                 font_color: Union[Tuple[int, int, int, int], Tuple[int, int, int], str] = "black",
                 hover_font_color: Optional[Union[Tuple[int, int, int, int], Tuple[int, int, int], str]] = None,
                 pressed_font_color: Optional[Union[Tuple[int, int, int, int], Tuple[int, int, int], str]] = None,
                 text_size: Union[int, str] = "20px",
                 hover_text_size: Optional[Union[int, str]] = None,
                 pressed_text_size: Optional[Union[int, str]] = None,
                 text_font: str = "Arial",
                 hover_text_font: Optional[str] = None,
                 pressed_text_font: Optional[str] = None,
                 background_color: Union[Tuple[int, int, int, int], Tuple[int, int, int], str] = "white",
                 hover_background_color: Optional[Union[Tuple[int, int, int, int], Tuple[int, int, int], str]] = None,
                 pressed_background_color: Optional[Union[Tuple[int, int, int, int], Tuple[int, int, int], str]] = None,
                 border_width: int = 0,
                 hover_border_width: Optional[int] = None,
                 pressed_border_width: Optional[int] = None,
                 border_color: Union[Tuple[int, int, int, int], Tuple[int, int, int], str] = "black",
                 hover_border_color: Optional[Union[Tuple[int, int, int, int], Tuple[int, int, int], str]] = None,
                 pressed_border_color: Optional[Union[Tuple[int, int, int, int], Tuple[int, int, int], str]] = None,
                 ) -> None:
        self.id = id
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height

        self.hover_text = hover_text or text
        self.pressed_text = pressed_text or hover_text

        self.text_align_x = text_align_x
        self.text_align_y = text_align_y
        self.position = position

        self.font_color = font_color
        self.hover_font_color = hover_font_color or font_color
        self.pressed_font_color = pressed_font_color or hover_font_color

        self.text_size = text_size
        self.hover_text_size = hover_text_size or text_size
        self.pressed_text_size = pressed_text_size or hover_text_size

        self.text_font = text_font
        self.hover_text_font = hover_text_font or text_font
        self.pressed_text_font = pressed_text_font or hover_text_font

        self.background_color = background_color
        self.hover_background_color = hover_background_color or background_color
        self.pressed_background_color = pressed_background_color or hover_background_color

        self.border_width = border_width
        self.hover_border_width = hover_border_width or border_width
        self.pressed_border_width = pressed_border_width or hover_border_width

        self.border_color = border_color
        self.hover_border_color = hover_border_color or border_color
        self.pressed_border_color = pressed_border_color or hover_border_color

