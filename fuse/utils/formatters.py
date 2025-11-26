import re


def format_size(b: int | float, d: int = 0) -> str:
    """Returns a formatted string of `b`"""

    for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
        if b < 1024:
            return f"{b:.{d}f}{unit}"
        b /= 1024

    return f"{b:.{d}f}EB"


def parse_size(size: str) -> int:
    """Returns bytes of a formatted string (e.g., `3.14MB`)"""
    size = size.strip().upper()
    units = {
        None: 1,
        "B": 1,
        "KB": 1024,
        "MB": 1024**2,
        "GB": 1024**3,
        "TB": 1024**4,
        "PB": 1024**5,
    }

    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGTPE]?B)?", size)
    if not m:
        raise ValueError(f"invalid format: {size}")

    number_str, unit_str = m.groups()
    num = float(number_str)

    return int(num * units[unit_str])


def format_time(seconds: float) -> str:
    """Transforms `seconds` into a formatted string"""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h} hours, {m} minutes, {s} seconds"
    elif m > 0:
        return f"{m} minutes, {s} seconds"
    else:
        return f"{s} seconds"
