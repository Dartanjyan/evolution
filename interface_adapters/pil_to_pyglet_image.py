import io

import pyglet
from PIL import Image


def pil_to_pyglet_image(pil_image: Image) -> pyglet.image.AbstractImage:
    """
    Конвертирует PIL.Image в pyglet.image.AbstractImage без сохранения на диск.

    :param pil_image: Объект PIL.Image.
    :return: Объект pyglet.image.AbstractImage.
    """
    # Преобразуем PIL.Image в байтовый буфер
    with io.BytesIO() as buffer:
        pil_image.save(buffer, format='PNG')  # Можно выбрать другой формат
        buffer.seek(0)  # Перематываем на начало буфера
        # Загружаем байты в pyglet
        pyglet_image = pyglet.image.load('',
                                         file=buffer)
    return pyglet_image