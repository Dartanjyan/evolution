# use_cases/update_label.py

import random

def get_random_text() -> str:
    """Генерирует случайное слово."""
    words = ["Hello", "World", "Clean", "Architecture", "Mouse", "Click"]
    return random.choice(words)
