from entities.gui.appstate import AppState


def switch_layout(state: AppState, new_layout: str):
    state.__current_layout = new_layout