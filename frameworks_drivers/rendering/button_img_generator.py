from typing import Union, Tuple
from PIL import Image, ImageFont, ImageDraw


def generate_image(text: str,
                    font_size: int,
                    width: int,
                    height: int,
                    font_name: str = "arial.ttf",
                    font_color: Union[str, float, Tuple[int]] = "black",
                    background_color: Union[str, float, Tuple[int]] = "white"
                    ) -> Image.Image:
    """
    This function was created generally for the pyglet buttons

    :param text: A text that is going to be written on the image
    :param font_size: Size of the font
    :param width: Width of the image
    :param height: Height of the image
    :param font_name: A .ttf path to the font, default is "arial.ttf"
    :param font_color: Font color
    :param background_color: Font background color
    :return: Image with a text in the center
    """
    if not font_name.endswith(".ttf"):
        font_name = font_name + ".ttf"

    font = ImageFont.truetype(font_name, font_size)
    image = Image.new("RGB", (width, height), background_color)
    draw = ImageDraw.Draw(image)
    text_width = draw.textlength(text, font)
    text_height = font_size
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    draw.text((x, y), text, font=font, fill=font_color)

    return image