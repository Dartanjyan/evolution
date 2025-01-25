from entities.gui.appstate import AppState
from entities.gui.layouts import GuiLayouts


def switch_layout(state: AppState, new_layout: GuiLayouts):
    state.current_layout = new_layout
