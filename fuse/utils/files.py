import sys

from typing import Iterator, IO, Any
from contextlib import contextmanager

from fuse.logger import log
from fuse.compression import CompressionFormat, compressed_file_writer


@contextmanager
def fuse_open(
    file: str | None,
    *args: Any,
    compression: CompressionFormat | None = None,
    compresslevel: int | None = None,
    **kwargs: Any,
) -> Iterator[IO[Any] | None]:
    """
    Context manager for opening files with optional compression.

    Opens files for output with support for compression formats ("gzip", "bzip2", "lzma").
    If no file is specified, returns `sys.stdout`. Handles errors gracefully.
    Returns `None` only if an error occurs.
    """
    fp: Any = None

    if file is None:
        yield sys.stdout
    elif compression is not None:
        with compressed_file_writer(
            file, compression, compresslevel=compresslevel
        ) as fp:
            yield fp
    else:
        try:
            fp = open(file, *args, **kwargs)
            yield fp
        except FileNotFoundError:
            log.error(f'file "{file}" not found.')
            yield None
        except PermissionError:
            log.error(f'no permission for "{file}".')
            yield None
        except IsADirectoryError:
            log.error(f'"{file}" is a directory.')
            yield None
        except FileExistsError:
            log.error(f'"{file}" already exists.')
            yield None
        except Exception as e:
            log.exception(f"unexpected error: {e}.")
            yield None
        finally:
            if fp is not None:
                fp.close()
