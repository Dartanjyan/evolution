from typing import Callable, Any

import pyglet.app

from entities.gui.appstate import AppState
from entities.gui.layouts import GuiLayouts


def get_button_handler(but_id: str, app_state: AppState, update_func: Callable[[], Any]) -> Callable[[pyglet.gui.WidgetBase], Any]:
    def update_decorator(func):
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            update_func()
        return wrapper

    def exit_app(widget):
        pyglet.app.exit()

    @update_decorator
    def switch_to_settings(widget):
        app_state.set_layout(GuiLayouts.APP_SETTINGS)

    @update_decorator
    def switch_to_editor(widget):
        app_state.set_layout(GuiLayouts.CREATURE_EDITOR)

    @update_decorator
    def switch_to_saves_list(widget):
        app_state.set_layout(GuiLayouts.LOAD_SIMULATION)

    ids = {
        'exit': exit_app,
        'settings': switch_to_settings,
        'editor': switch_to_editor,
        'load': switch_to_saves_list
    }

    return ids[but_id]