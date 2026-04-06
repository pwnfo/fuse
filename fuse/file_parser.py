import re

from pathlib import Path
from typing import Iterator

from fuse.logger import log
from fuse.utils.files import secure_open


class InvalidSyntaxError(Exception):
    """Raised on invalid fuse file syntax."""

    def __init__(self, message: str) -> None:
        super().__init__("invalid file: " + message)


def process_expr_file(
    filepath: str,
) -> Iterator[tuple[str, list[str]]]:
    with secure_open(filepath, "r", encoding="utf-8") as fp:
        if fp is None:
            return

        lines = [line.strip() for line in fp if line.strip()]

    aliases: list[tuple[str, str]] = []
    current_files: list[str] = []

    log.info(f"opening file '{filepath}' (with {len(lines)} lines).")

    for i, line in enumerate(lines):
        # expand aliases
        for alias_key, alias_val in aliases:
            line = re.sub(r"(?<!\\)\$" + re.escape(alias_key) + ";", alias_val, line)

        fields = line.split(" ")
        keyword = fields[0]
        arguments = fields[1:]

        # comment
        if keyword == "#":
            continue

        # alias definition
        if keyword == r"%alias":
            if len(fields) < 3:
                raise InvalidSyntaxError("alias keyword requires 2 arguments.")
            a_name = arguments[0].strip()
            a_value = " ".join(arguments[1:])
            if ";" in a_name or "$" in a_name:
                raise InvalidSyntaxError("alias name cannot contain ';' or '$'.")
            aliases.append((a_name, a_value))
            continue

        # file include
        if keyword == r"%file":
            if len(fields) < 2:
                raise InvalidSyntaxError("'%file' keyword requires 1 argument.")

            if arguments[0].startswith("./"):
                base_dir = Path(Path(filepath).resolve()).parent
                file = str((base_dir / " ".join(arguments).strip()).resolve())
            else:
                file = " ".join(arguments).strip()

            current_files.append(file)
            continue

        # expression line
        yield line, list(current_files) if current_files else []
        current_files = []
