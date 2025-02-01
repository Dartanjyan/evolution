def procent_to_px(from_val: str | None, window_height: int) -> int:
    if from_val.endswith("px") or from_val.isnumeric():
        pixels = int(from_val.replace("px", ""))
    elif from_val.endswith("%"):
        pixels = int(float(from_val.replace("%", "")) / 100 * window_height)
    else:
        pixels = 16
    return max(pixels, 1)