def procent_to_px(from_val: str, window_height: int) -> int:
    if from_val.endswith("px") or from_val.isnumeric():
        font_size = int(from_val.replace("px", ""))
    elif from_val.endswith("%"):
        font_size = int(float(from_val.replace("%", "")) / 100 * window_height)
    else:
        font_size = 16
    return max(font_size, 1)