import re

char_classes = [
    ["d", "0123456789"],
    ["h", "0123456789abcdef"],
    ["H", "0123456789ABCDEF"],
    ["a", "abcdefghijklmnopqrstuvwxyz"],
    ["A", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
    ["s", " "],
    ["o", "01234567"],
    ["p", "!@#$%^&*()-_+="],
    ["l", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"],
    ["N", "\n"],
]


def pattern_repl(pattern: str, wc: str = "/") -> str:
    """Replaces the word classes of an `pattern`"""

    def i_replace(m: re.Match) -> str:
        """REGEX function
        "[/d]" -> "[0123456789]"
        "/d"   -> "[0123456789]"
        """
        expr = m.group(0)
        if expr.startswith("["):
            return expr.replace(i_old, i_new)
        return expr.replace(i_old, f"[{i_new}]")

    for _ in char_classes:
        i_old = wc + _[0]
        i_new = _[1]
        pattern = re.sub(r"\[[^\]]*\]|[^[]+", i_replace, pattern)

    return pattern
