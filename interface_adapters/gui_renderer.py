from entities.gui.layouts import GuiLayouts

class GuiRenderer:
    def __init__(self, switch_layout_callback):
        self.switch_layout_callback = switch_layout_callback
        self.current_layout = None
        self.buttons = []

    def set_layout(self, layout: GuiLayouts):
        self.current_layout = layout
        self.buttons = self._get_buttons_for_layout(layout)

    def _get_buttons_for_layout(self, layout: GuiLayouts):
        if layout == GuiLayouts.MAIN_MENU:
            return [
                {"label": "Settings", "x": 300, "y": 400, "callback": lambda: self.switch_layout_callback(GuiLayouts.SETTINGS)},
                {"label": "Simulation Editor", "x": 300, "y": 300, "callback": lambda: self.switch_layout_callback(GuiLayouts.SIMULATION_EDITOR)}
            ]
        elif layout == GuiLayouts.SETTINGS:
            return [
                {"label": "Back to Main Menu", "x": 300, "y": 400, "callback": lambda: self.switch_layout_callback(GuiLayouts.MAIN_MENU)}
            ]
        elif layout == GuiLayouts.SIMULATION_EDITOR:
            return [
                {"label": "Back to Main Menu", "x": 300, "y": 400, "callback": lambda: self.switch_layout_callback(GuiLayouts.MAIN_MENU)}
            ]
        return []

    def get_render_data(self):
        """Возвращает информацию о текущих кнопках для рендера."""
        return self.buttons

    def handle_mouse_press(self, x, y):
        """Обрабатывает нажатия мыши и вызывает соответствующие колбэки."""
        for button in self.buttons:
            bx, by = button["x"], button["y"]
            if bx - 50 <= x <= bx + 50 and by - 20 <= y <= by + 20:
                button["callback"]()
