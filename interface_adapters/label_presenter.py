# interface_adapters/label_presenter.py

from entities.label import get_default_label_text
from use_cases.gui.update_label import get_random_text

class LabelPresenter:
    def __init__(self):
        self.text = get_default_label_text()

    def get_label(self) -> str:
        """Получить текущий текст."""
        return self.text

    def update_label(self):
        """Обновить текст случайным словом."""
        self.text = get_random_text()
