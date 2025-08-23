import sys

from contextlib import contextmanager
from typing import Any, Iterator, IO

from ..console import log


@contextmanager
def r_open(file: str | None, *args: Any, **kwargs: Any) -> Iterator[IO[Any] | None]:
    """Opens file and handles possible errors. Returns `sys.stdout` if `file=None`"""
    if file is None:
        yield sys.stdout
    else:
        try:
            fp = open(file, *args, **kwargs)
            yield fp
        except FileNotFoundError:
            log.error(f'file "{file}" not found.')
            yield None
        except PermissionError:
            log.error(f'no read permission for "{file}".')
            yield None
        except IsADirectoryError:
            log.error(f'"{file}" is a directory.')
            yield None
        except Exception as e:
            log.exception(f"unexpected error: {e}.")
            yield None
        finally:
            try:
                fp.close()
            except Exception:
                pass


def format_size(b: int | float, d: int = 0) -> str:
    """Returns a formatted string of `b`"""
    try:
        for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
            if b < 1024:
                return f"{b:.{d}f}{unit}"
            b /= 1024
        return f"{b:.{d}f}EB"
    except OverflowError:
        return "<OverflowError>"


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
