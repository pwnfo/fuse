import re

from pathlib import Path
from typing import Iterator

from fuse.utils.files import fuse_open


class InvalidSyntaxError(Exception):
    """Raised on invalid fuse file syntax."""

    def __init__(self, message: str) -> None:
        super().__init__("invalid file: " + message)


def process_expr_file(
    filepath: str,
) -> Iterator[tuple[str, list[str]]]:
    """
    Yields the lines of the expression file (`filepath`) to be processed
    with included files. Handles keywords (`%define` and `%include`).
    """
    with fuse_open(filepath, "r", encoding="utf-8") as fp:
        if fp is None:
            return

        lines = [line.strip() for line in fp if line.strip()]

    defines: list[tuple[str, str]] = []
    current_files: list[str] = []

    for i, line in enumerate(lines):
        # expand defines
        for def_key, def_val in defines:
            line = re.sub(r"(?<!\\)\$" + re.escape(def_key) + ";", def_val, line)

        fields = line.split(" ")
        keyword = fields[0]
        arguments = fields[1:]

        # comment
        if keyword == "#":
            continue

        # ´define´ definition
        if keyword == r"%define":
            if len(fields) < 3:
                raise InvalidSyntaxError("'%define' keyword requires 2 arguments.")
            d_name = arguments[0].strip()
            d_value = " ".join(arguments[1:])
            if ";" in d_name or "$" in d_name:
                raise InvalidSyntaxError("define name cannot contain ';' or '$'.")
            defines.append((d_name, d_value))
            continue

        # ´include´ definition
        if keyword == r"%include":
            if len(fields) < 2:
                raise InvalidSyntaxError("'%include' keyword requires 1 argument.")

            # gets inclusions relative to the expression file
            # if the path starts with "./" or "../".
            if arguments[0].startswith("./") or arguments[0].startswith("../"):
                base_dir = Path(filepath).resolve().parent
                file = str((base_dir / " ".join(arguments).strip()).resolve())
            else:
                file = " ".join(arguments).strip()

            current_files.append(file)
            continue

        # expression line
        yield line, list(current_files) if current_files else []
        current_files = []
