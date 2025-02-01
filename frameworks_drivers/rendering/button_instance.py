import os.path

import pyglet.graphics

from entities.gui.widgets import PushButton
from frameworks_drivers.rendering.button_img_generator import generate_image
from interface_adapters.pil_to_pyglet_image import pil_to_pyglet_image

def get_button(button: PushButton, batch: pyglet.graphics.Batch, font_folder:str = "") -> pyglet.gui.PushButton:
    unpressed = generate_image(
        text=button.text,
        font_size=button.text_size,
        width=button.width,
        height=button.height,
        font_name=os.path.join(font_folder, button.text_font),
        font_color=button.font_color,
        background_color=button.background_color,
        border_width=button.border_width
    )
    hover = generate_image(
        text=button.hover_text,
        font_size=button.hover_text_size,
        width=button.width,
        height=button.height,
        font_name=os.path.join(font_folder, button.hover_text_font),
        font_color=button.hover_font_color,
        background_color=button.hover_background_color,
        border_width=button.hover_border_width
    )
    pressed = generate_image(
        text=button.pressed_text,
        font_size=button.pressed_text_size,
        width=button.width,
        height=button.height,
        font_name=os.path.join(font_folder, button.pressed_text_font),
        font_color=button.pressed_font_color,
        background_color=button.pressed_background_color,
        border_width=button.pressed_border_width
    )
    button = pyglet.gui.PushButton(
        x=button.x,
        y=button.y,
        pressed=pil_to_pyglet_image(pressed),
        unpressed=pil_to_pyglet_image(unpressed),
        hover=pil_to_pyglet_image(hover),
        batch=batch
    )

    return button
