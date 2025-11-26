import sys

from typing import Iterator, IO, Any
from contextlib import contextmanager

from fuse.logger import log


@contextmanager
def secure_open(
    file: str | None, *args: Any, **kwargs: Any
) -> Iterator[IO[Any] | None]:
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
                fp.close()  # type: ignore
            except Exception:
                pass
