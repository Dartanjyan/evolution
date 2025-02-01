from entities.gui.widgets import PushButton, Position
from use_cases.gui.font_size_from_config import procent_to_px


def push_button(item: dict, height: int) -> PushButton:
    def safe_get(key: str | int, _item: dict = item):
        return _item[key] if key in _item.keys() else None

    pos = safe_get("position")

    position = Position(
        x=safe_get("x", pos),
        y=safe_get("y", pos),
        anchor_x=safe_get("anchor-x", pos),
        anchor_y=safe_get("anchor-y", pos)
    )
    btn = PushButton(
        item["id"],
        item["text"],
        procent_to_px(pos["x"], height),
        procent_to_px(pos["y"], height),
        procent_to_px(safe_get("width"), height),
        procent_to_px(safe_get("height"), height),
        safe_get("hover-text"),
        safe_get("pressed-text"),
        safe_get("text-align-x"),
        safe_get("text-align-y"),
        position,
        safe_get("font-color"),
        safe_get("hover-font-color"),
        safe_get("pressed-font-color"),
        procent_to_px(safe_get("text-size"), height) if safe_get("text-size") else None,
        procent_to_px(safe_get("hover-text-size"), height) if safe_get("hover-text-size") else None,
        procent_to_px(safe_get("pressed-text-size"), height) if safe_get("pressed-text-size") else None,
        safe_get("text-font"),
        safe_get("hover-text-font"),
        safe_get("pressed-text-font"),
        safe_get("background-color"),
        safe_get("hover-background-color"),
        safe_get("pressed-background-color"),
        procent_to_px(safe_get("border-width"), height) if safe_get("border-width") else None,
        procent_to_px(safe_get("hover-border-width"), height) if safe_get("hover-border-width") else None,
        procent_to_px(safe_get("pressed-border-width"), height) if safe_get("pressed-border-width") else None,
        safe_get("border-color"),
        safe_get("hover-border-color"),
        safe_get("pressed-border-color")
    )
    return btn

