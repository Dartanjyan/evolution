from entities.gui.layouts import GuiLayouts


class AppState:
    __current_layout: GuiLayouts
    def __init__(self, current_layout: GuiLayouts) -> None:
        self.__current_layout = current_layout

    def get_layout(self) -> GuiLayouts:
        return self.__current_layout
