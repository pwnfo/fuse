import os
import sys
import logging

from typing import Any

environ_nocolor = os.environ.get("FUSE_NOCOLOR")
FUSE_NOCOLOR = environ_nocolor is not None and environ_nocolor != "0"

if not FUSE_NOCOLOR:
    from rich.console import Console
    from rich.logging import RichHandler

    class FuseRichHandler(RichHandler):
        fuse_console = Console(file=sys.stdout)
        fuse_err_console = Console(file=sys.stderr)

        def emit(self, record: logging.LogRecord) -> Any:

            # errors to stderr
            if record.levelno < logging.WARNING:
                self.console = self.fuse_console
            else:
                self.console = self.fuse_err_console
            super().emit(record)

    console = Console()


class FuseStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> Any:
        if record.levelno < logging.WARNING:
            self.stream = sys.stdout
        else:
            self.stream = sys.stderr
        super().emit(record)


def setup_logger() -> logging.Logger:
    log = logging.getLogger(__name__)
    handler: FuseStreamHandler | FuseRichHandler

    if FUSE_NOCOLOR:
        handler = FuseStreamHandler(sys.stdout)
    else:
        handler = FuseRichHandler(
            markup=True,
            show_time=False,
            show_level=False,
            show_path=False,
        )

    log.setLevel(logging.INFO)
    log.addHandler(handler)
    log.propagate = False

    return log


log = setup_logger()


if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    try:
        1 / 0  # type: ignore
    except:
        log.exception("Exception...")
    log.debug("Debug...")
    log.warning("Warning...")
    log.info("Info...")
    log.error("Error...")
    log.critical("Criticial...")

    log.info("Info with\nBreak line...")
