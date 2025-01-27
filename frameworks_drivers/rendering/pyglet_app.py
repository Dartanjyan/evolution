# frameworks_drivers/pyglet_renderer.py
import os

import pyglet

from typing import List

from entities.gui.appstate import AppState
from entities.gui.font_size_from_config import procent_to_px
from entities.gui.layouts import GuiLayouts
from frameworks_drivers.database.json_reader import JsonReader
from frameworks_drivers.rendering.button_img_generator import generate_image
from interface_adapters.label_presenter import LabelPresenter
from interface_adapters.pil_to_pyglet_image import pil_to_pyglet_image
from use_cases.gui.layout_manager import LayoutGetter


class PygletApp:
    def __init__(self):
        self.presenter = LabelPresenter()
        self.window = pyglet.window.Window(width=800, height=600)
        self.label = None
        self.batch = pyglet.graphics.Batch()
        self.all_gui_items = []

        self.layout_getter = LayoutGetter(JsonReader())

        self.current_layout: AppState = AppState(GuiLayouts.MAIN_MENU)

    def get_layout(self, layout: GuiLayouts) -> dict:
        return self.layout_getter.get_layout(layout.value)

    def setup(self):
        """Подготовить интерфейс для отображения."""

        main_menu_layout = self.get_layout(self.current_layout.get_layout())

        if title := main_menu_layout["title"]:
            self.window.set_caption(f"Evolution: {title}")
        if items := main_menu_layout["items"]:
            for item in items:
                print(item)

                font_size = None
                pos = item["position"]

                if item["text"]:
                    font_size = procent_to_px(item["text-size"], self.window.height)
                if item["type"] == "label":
                    label = pyglet.text.Label(text=item["text"],
                                              font_name=item["text-font"],
                                              font_size=font_size,
                                              x=procent_to_px(pos["x"], self.window.width),
                                              y=procent_to_px(pos["y"], self.window.height),
                                              anchor_x=pos["anchor-x"], anchor_y=pos["anchor-x"],
                                              batch=self.batch)
                    self.all_gui_items.append(label)
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
                        background_color="grey"
                    )
                    pressed = generate_image(
                        text=item["text"],
                        font_size=procent_to_px(item["text-size"], self.window.height),
                        width=procent_to_px(item["width"], self.window.width),
                        height=procent_to_px(item["height"], self.window.height),
                        font_name=os.path.join(font_path, item["text-font"] + "_Bold"),
                        font_color="black",
                        background_color="grey"
                    )

                    button = pyglet.gui.PushButton(
                        x=procent_to_px(pos["x"], self.window.width) - (procent_to_px(item["width"], self.window.width)//2),
                        y=procent_to_px(pos["y"], self.window.height) - (procent_to_px(item["height"], self.window.height)//2),
                        pressed=pil_to_pyglet_image(pressed),
                        unpressed=pil_to_pyglet_image(unpressed),
                        hover=pil_to_pyglet_image(hover),
                        batch=self.batch
                    )
                    button.push_handlers(self.window)
                    self.all_gui_items.append(button)

    def update_label(self):
        """Обновить текст метки."""
        self.presenter.update_label()
        self.label.text = self.presenter.get_label()

    def run(self):
        """Запустить графический интерфейс."""

        @self.window.event
        def on_draw():
            self.window.clear()
            self.batch.draw()

        self.setup()
        pyglet.app.run()
