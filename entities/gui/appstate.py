from dataclasses import dataclass

from entities.gui.layouts import GuiLayouts


@dataclass
class AppState:
    current_layout: GuiLayouts
