from entities import config
from entities.gui.appstate import AppState
from entities.gui.layouts import GuiLayouts
from interface_adapters.json_reader_interface import JsonReaderInterface


def switch_layout(state: AppState, new_layout: GuiLayouts):
    state.__current_layout = new_layout


class LayoutGetter:
    def __init__(self, json_reader: JsonReaderInterface):
        self.json_reader = json_reader

    def get_layout(self, layout_name: str) -> dict:
        data = self.json_reader.read_json(config.LAYOUT_PATH)
        return data.get(layout_name, {})
