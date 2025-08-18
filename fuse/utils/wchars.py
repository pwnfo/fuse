import re

wchars = [
    ["d", "0123456789"],
    ["h", "0123456789abcdef"],
    ["H", "0123456789ABCDEF"],
    ["a", "abcdefghijklmnopqrstuvwxyz"],
    ["A", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
    ["s", " "],
    ["o", "01234567"],
    ["p", "!@#$%^&*()-_+="],
    ["l", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"],
]


def pattern_repl(pattern: str) -> str:
    for x in wchars:
        pattern = re.sub(rf"(?<!\\)\/{x[0]}", f"[{x[1]}]", pattern)
        pattern = re.sub(rf"(?<!\\)\*{x[0]}", x[1], pattern)

    return pattern
