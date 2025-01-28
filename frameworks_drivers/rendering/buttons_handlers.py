from typing import Callable, Any

import pyglet.app

from entities.gui.appstate import AppState
from entities.gui.layouts import GuiLayouts


def get_button_handler(but_id: str, app_state: AppState, update_func: Callable[[], Any]) -> Callable[[pyglet.gui.WidgetBase], Any]:
    def exit_app(widget):
        pyglet.app.exit()

    def switch_to_settings(widget):
        app_state.set_layout(GuiLayouts.APP_SETTINGS)
        update_func()

    def switch_to_editor(widget):
        app_state.set_layout(GuiLayouts.CREATURE_EDITOR)
        update_func()

    def switch_to_saves_list(widget):
        app_state.set_layout(GuiLayouts.LOAD_SIMULATION)
        update_func()

    def switch_to_main_menu(widget):
        app_state.set_layout(GuiLayouts.MAIN_MENU)
        update_func()

    ids = {
        'settings': switch_to_settings,
        'editor': switch_to_editor,
        'load': switch_to_saves_list,
        'main-menu': switch_to_main_menu,
        'exit': exit_app
    }

    return ids[but_id]