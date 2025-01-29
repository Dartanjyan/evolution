# frameworks_drivers/pyglet_renderer.py
import os
from typing import List

import pyglet

from entities.gui.appstate import AppState
from frameworks_drivers.rendering.button_instance import get_button
from frameworks_drivers.rendering.buttons_handlers import get_button_handler
from use_cases.gui.font_size_from_config import procent_to_px
from entities.gui.layouts import GuiLayouts
from frameworks_drivers.database.json_reader import JsonReader
from use_cases.gui.layout_manager import LayoutGetter

class PygletApp:
    def __init__(self):
        self.window = pyglet.window.Window(width=800, height=600, resizable=True)
        self.window.set_minimum_size(320, 200)

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
                # print(item)

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

                    button = get_button(
                        item["text"],
                        procent_to_px(item["width"], self.window.width),
                        procent_to_px(item["height"], self.window.height),
                        procent_to_px(item["text-size"], self.window.height),
                        os.path.join(font_path, "Arial.ttf"),
                        os.path.join(font_path, "Arial_Bold.ttf"),
                        os.path.join(font_path, "Arial_Bold.ttf"),
                        procent_to_px(pos["x"], self.window.width) - (procent_to_px(item["width"], self.window.width) // 2),
                        procent_to_px(pos["y"], self.window.height) - (procent_to_px(item["height"], self.window.height) // 2),
                        self.gui_batch
                    )

                    self.window.push_handlers(button)
                    self.all_buttons.append(button)

                    has_no_id = False
                    try:
                        item["id"]
                    except KeyError:
                        has_no_id = True

                    try:
                        button.on_release = get_button_handler(item["id"], self.app_state, self.setup)
                    except KeyError:
                        button.on_release = lambda widget: print(f"This button does not have {'ID' if has_no_id else 'handler'}.")


    def run(self):
        """Запустить графический интерфейс."""

        @self.window.event
        def on_resize(width, height):
            self.setup()

        @self.window.event
        def on_draw():
            self.window.clear()
            self.gui_batch.draw()

        def update(dt):
            pass


        pyglet.clock.schedule_interval(update, 1/60)
        pyglet.app.run()
