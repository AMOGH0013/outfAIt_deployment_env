from __future__ import annotations

COLOR_BASE_ALIASES: dict[str, str] = {
    "baby pink": "pink",
    "blush pink": "pink",
    "burgundy": "red",
    "charcoal": "gray",
    "charcoal gray": "gray",
    "cream": "white",
    "dark denim": "blue",
    "dark gray": "gray",
    "dark green": "green",
    "dark navy": "navy",
    "dark purple": "purple",
    "dark red": "red",
    "dark teal": "teal",
    "denim blue": "blue",
    "dusty rose": "pink",
    "forest green": "green",
    "hot pink": "pink",
    "indigo": "blue",
    "khaki": "beige",
    "lavender": "purple",
    "light blue": "blue",
    "light gray": "gray",
    "light pink": "pink",
    "magenta": "pink",
    "maroon": "red",
    "mauve": "pink",
    "off white": "white",
    "olive green": "olive",
    "powder blue": "blue",
    "royal purple": "purple",
    "salmon": "pink",
    "sea green": "teal",
    "sky blue": "blue",
    "slate": "gray",
    "tan": "beige",
    "teal green": "teal",
    "violet": "purple",
}

NEUTRAL_COLOR_BASES = {"black", "white", "gray", "navy", "beige", "brown", "teal", "olive"}


def simplify_color_label(value: str | None) -> str:
    if not value:
        return ""
    return str(value).strip().replace("_", " ").lower()


def normalize_color_name(value: str | None) -> str:
    text = simplify_color_label(value)
    if not text:
        return ""
    if text in {"unknown", "multicolor"}:
        return text
    if "-" in text:
        text = text.split("-", 1)[0].strip()
    return COLOR_BASE_ALIASES.get(text, text)


def is_neutral_color(value: str | None) -> bool:
    return normalize_color_name(value) in NEUTRAL_COLOR_BASES
