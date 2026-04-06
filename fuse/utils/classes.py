CHAR_CLASSES: dict[str, str] = {
    "d": "0123456789",
    "h": "0123456789abcdef",
    "H": "0123456789ABCDEF",
    "a": "abcdefghijklmnopqrstuvwxyz",
    "A": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "s": " ",
    "o": "01234567",
    "p": "!@#$%^&*()-_+=",
    "l": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "N": "\n",
}


def pattern_repl(pattern: str, wc: str = "/") -> str:
    """Replace character-class shortcuts"""
    out: list[str] = []
    i = 0
    n = len(pattern)
    wc_len = len(wc)

    while i < n:
        ch = pattern[i]

        if ch == "\\":
            if i + 1 < n:
                out.append(ch)
                out.append(pattern[i + 1])
                i += 2
            else:
                out.append(ch)
                i += 1
            continue

        # literal group (...)
        if ch == "(":
            out.append(ch)
            i += 1
            while i < n:
                if pattern[i] == "\\" and i + 1 < n:
                    out.append(pattern[i])
                    out.append(pattern[i + 1])
                    i += 2
                elif pattern[i] == ")":
                    out.append(pattern[i])
                    i += 1
                    break
                else:
                    out.append(pattern[i])
                    i += 1
            continue

        # bracketed class [...]
        if ch == "[":
            out.append(ch)
            i += 1
            while i < n:
                if pattern[i] == "\\" and i + 1 < n:
                    out.append(pattern[i])
                    out.append(pattern[i + 1])
                    i += 2
                elif pattern[i] == "]":
                    out.append(pattern[i])
                    i += 1
                    break
                elif pattern[i : i + wc_len] == wc and i + wc_len < n:
                    key = pattern[i + wc_len]
                    if key in CHAR_CLASSES:
                        out.append(CHAR_CLASSES[key])
                        i += wc_len + 1
                    else:
                        out.append(pattern[i])
                        i += 1
                else:
                    out.append(pattern[i])
                    i += 1
            continue

        # outside brackets
        if pattern[i : i + wc_len] == wc and i + wc_len < n:
            key = pattern[i + wc_len]
            if key in CHAR_CLASSES:
                out.append("[")
                out.append(CHAR_CLASSES[key])
                out.append("]")
                i += wc_len + 1
                continue

        out.append(ch)
        i += 1

    return "".join(out)
