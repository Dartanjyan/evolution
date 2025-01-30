import pyglet.graphics

from frameworks_drivers.rendering.button_img_generator import generate_image
from interface_adapters.pil_to_pyglet_image import pil_to_pyglet_image


def get_button(text: str, w_px: int, h_px: int, font_size: int, unpressed_font: str, hover_font: str, pressed_font: str,
               but_x: int, but_y: int, batch: pyglet.graphics.Batch, border_width: int = 2):
    unpressed = generate_image(
        text=text,
        font_size=font_size,
        width=w_px,
        height=h_px,
        font_name=unpressed_font,
        font_color="black",
        background_color="white",
        border_width=border_width
    )
    hover = generate_image(
        text=text,
        font_size=font_size,
        width=w_px,
        height=h_px,
        font_name=hover_font,
        font_color="black",
        background_color="white",
        border_width=border_width*2
    )
    pressed = generate_image(
        text=text,
        font_size=font_size,
        width=w_px,
        height=h_px,
        font_name=pressed_font,
        font_color="black",
        background_color="lightgray",
        border_width=border_width*4
    )

    button = pyglet.gui.PushButton(
        x=but_x,
        y=but_y,
        pressed=pil_to_pyglet_image(pressed),
        unpressed=pil_to_pyglet_image(unpressed),
        hover=pil_to_pyglet_image(hover),
        batch=batch
    )

    # h_px = procent_to_px(item["height"], self.window.height)
    # os.path.join(font_path, item["text-font"])
    #
    # button
    # x=procent_to_px(pos["x"], self.window.width) - (w_px // 2),

    return button