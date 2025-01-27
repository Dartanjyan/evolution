# frameworks_drivers/pyglet_renderer.py
import os
from typing import List

import pyglet

from entities.gui.appstate import AppState
from frameworks_drivers.rendering.buttons_handlers import get_button_handler
from use_cases.gui.font_size_from_config import procent_to_px
from entities.gui.layouts import GuiLayouts
from frameworks_drivers.database.json_reader import JsonReader
from frameworks_drivers.rendering.button_img_generator import generate_image
from interface_adapters.pil_to_pyglet_image import pil_to_pyglet_image
from use_cases.gui.layout_manager import LayoutGetter


def exit_app(widget):
    pyglet.app.exit()


class PygletApp:
    def __init__(self):
        self.window = pyglet.window.Window(width=800, height=600, resizable=True)
        self.gui_batch = pyglet.graphics.Batch()
        self.all_labels: List[pyglet.text.Label] = []
        self.all_buttons: List[pyglet.gui.PushButton] = []

        self.layout_getter = LayoutGetter(JsonReader())

        self.app_state: AppState = AppState(GuiLayouts.MAIN_MENU)

    def get_layout(self, layout: GuiLayouts) -> dict:
        return self.layout_getter.get_layout(layout.value)

    def setup(self):
        """Подготовить интерфейс для отображения."""

        self.gui_batch = pyglet.graphics.Batch()
        self.all_labels: List[pyglet.text.Label] = []
        self.all_buttons: List[pyglet.gui.PushButton] = []

        main_menu_layout = self.get_layout(self.app_state.get_layout())

        if title := main_menu_layout["title"]:
            self.window.set_caption(f"Evolution: {title}")
        if items := main_menu_layout["items"]:
            for item in items:
                print(item)

                font_size = None
                pos = item["position"]
                try:
                    color = item["font-color"]
                except KeyError:
                    color = (255, 255, 255, 255)

                if item["text"]:
                    font_size = procent_to_px(item["text-size"], self.window.height)
                if item["type"] == "label":
                    label = pyglet.text.Label(text=item["text"],
                                              font_name=item["text-font"],
                                              font_size=font_size,
                                              x=procent_to_px(pos["x"], self.window.width),
                                              y=procent_to_px(pos["y"], self.window.height),
                                              anchor_x=pos["anchor-x"], anchor_y=pos["anchor-y"],
                                              color=color,
                                              batch=self.gui_batch)
                    self.all_labels.append(label)
                elif item["type"] == "press-button":
                    font_path = "resources"
                    unpressed = generate_image(
                        text=item["text"],
                        font_size=procent_to_px(item["text-size"], self.window.height),
                        width=procent_to_px(item["width"], self.window.width),
                        height=procent_to_px(item["height"], self.window.height),
                        font_name=os.path.join(font_path, item["text-font"]),
                        font_color="black",
                        background_color="white"
                    )
                    hover = generate_image(
                        text=item["text"],
                        font_size=procent_to_px(item["text-size"], self.window.height),
                        width=procent_to_px(item["width"], self.window.width),
                        height=procent_to_px(item["height"], self.window.height),
                        font_name=os.path.join(font_path, item["text-font"] + "_Bold"),
                        font_color="black",
                        background_color="white"
                    )
                    pressed = generate_image(
                        text=item["text"],
                        font_size=procent_to_px(item["text-size"], self.window.height),
                        width=procent_to_px(item["width"], self.window.width),
                        height=procent_to_px(item["height"], self.window.height),
                        font_name=os.path.join(font_path, item["text-font"] + "_Bold"),
                        font_color="black",
                        background_color="lightgray"
                    )

                    button = pyglet.gui.PushButton(
                        x=procent_to_px(pos["x"], self.window.width) - (procent_to_px(item["width"], self.window.width)//2),
                        y=procent_to_px(pos["y"], self.window.height) - (procent_to_px(item["height"], self.window.height)//2),
                        pressed=pil_to_pyglet_image(pressed),
                        unpressed=pil_to_pyglet_image(unpressed),
                        hover=pil_to_pyglet_image(hover),
                        batch=self.gui_batch
                    )

                    try:
                        button.on_press = get_button_handler(item["id"], self.app_state, self.setup)
                    except KeyError:
                        button.on_press = lambda widget: print(f"This button does not have handler.")

                    if item["id"] == "exit":
                        button.on_press = exit_app

                    self.window.push_handlers(button)
                    self.all_buttons.append(button)

    def run(self):
        """Запустить графический интерфейс."""

        @self.window.event
        def on_resize(width, height):
            self.setup()

        @self.window.event
        def on_draw():
            self.window.clear()
            self.gui_batch.draw()

        self.setup()
        pyglet.app.run(1/30)
