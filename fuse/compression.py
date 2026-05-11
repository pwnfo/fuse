import gzip
import bz2
import lzma

from typing import IO, Any, Iterator, Literal
from contextlib import contextmanager

from fuse.logger import log

CompressionFormat = Literal["gzip", "bzip2", "lzma"]

COMPRESSION_EXTENSIONS = {
    "gzip": ".gz",
    "bzip2": ".bz2",
    "lzma": ".xz",
}

COMPRESSION_LEVEL_RANGES = {"gzip": (0, 9), "bzip2": (1, 9), "lzma": (0, 9)}

COMPRESSION_DEFAULT_LEVELS: dict[CompressionFormat, int] = {
    "gzip": 9,
    "bzip2": 9,
    "lzma": 6,
}

# COMPRESSION_ALIASES = {
#     "gz": "gzip",
#     "bz2": "bzip2",
#     "xz": "lzma",
# }


def get_compression_extension(format_name: CompressionFormat) -> str:
    """Get the file extension for a compression format."""
    return COMPRESSION_EXTENSIONS[format_name]


def ensure_compression_extension(compression: CompressionFormat, path: str) -> str:
    """
    Ensure path has the correct extension for the compression format.
    Path with the correct compression extension appended (if needed).
    """
    ext = get_compression_extension(compression)
    return path if path.endswith(ext) else path + ext


@contextmanager
def compressed_file_writer(
    file: str,
    compression: CompressionFormat,
    compresslevel: int | None = None,
    encoding: str = "utf-8",
) -> Iterator[IO[Any] | None]:
    """Context manager for writing compressed files."""
    fp: Any = None
    output_path = None

    try:
        # if compression is None:
        #     fp = open(file, "a", encoding=encoding, buffering=buffering)
        #     yield fp
        #     return

        output_path = ensure_compression_extension(compression, file)
        c_kwargs: dict[str, Any] = {"encoding": encoding}

        c_kwargs["compresslevel" if compression != "lzma" else "preset"] = (
            compresslevel
            if compresslevel is not None
            else COMPRESSION_DEFAULT_LEVELS[compression]
        )

        match compression:
            case "gzip":
                fp = gzip.open(output_path, "xt", **c_kwargs)
            case "bzip2":
                fp = bz2.open(output_path, "xt", **c_kwargs)
            case "lzma":
                fp = lzma.open(output_path, "xt", **c_kwargs)
            case _:
                log.error(f"unsupported compression format: '{compression}'.")
                yield None

        yield fp

    except FileNotFoundError:
        log.error(f'file "{output_path or file}" not found.')
        yield None
    except PermissionError:
        log.error(f'no permission for "{output_path or file}".')
        yield None
    except IsADirectoryError:
        log.error(f'"{output_path or file}" is a directory.')
        yield None
    except FileExistsError:
        log.error(f'"{output_path or file}" already exists.')
        yield None
    except Exception as e:
        log.exception(f"unexpected error: {e}.")
        yield None
    finally:
        if fp is not None:
            fp.close()
