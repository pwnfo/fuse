import re

CHAR_CLASSES = (
    ("d", "0123456789"),
    ("h", "0123456789abcdef"),
    ("H", "0123456789ABCDEF"),
    ("a", "abcdefghijklmnopqrstuvwxyz"),
    ("A", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    ("s", " "),
    ("o", "01234567"),
    ("p", "!@#$%^&*()-_+="),
    ("l", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    ("N", "\n"),
)


def pattern_repl(pattern: str, wc: str = "/") -> str:
    """Replaces the word classes of an `pattern`."""

    for char_class in CHAR_CLASSES:
        i_old = wc + char_class[0]
        i_new = char_class[1]

        unescaped_i_old_re = re.compile(rf"(?<!\\){re.escape(i_old)}")

        def i_replace(m: re.Match) -> str:
            expr = m.group(0)
            if expr.startswith("["):
                return unescaped_i_old_re.sub(i_new, expr)

            if expr.startswith("("):
                return expr

            return unescaped_i_old_re.sub(lambda mo: f"[{i_new}]", expr)

        pattern = re.sub(r"(?<!\\)\[[^\]]*\]|(?<!\\)[^[]+", i_replace, pattern)

    return pattern
