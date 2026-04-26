import sys
import logging

from typing import Any


class FuseFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.WARNING:
            return f"Warning :: {record.getMessage()}"
        return record.getMessage()


class FuseStreamHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> Any:
        if record.levelno < logging.WARNING:
            self.stream = sys.stdout
        else:
            self.stream = sys.stderr
        super().emit(record)


def setup_logger() -> logging.Logger:
    log = logging.getLogger(__name__)

    handler = FuseStreamHandler(sys.stdout)
    handler.setFormatter(FuseFormatter())

    log.setLevel(logging.INFO)
    log.addHandler(handler)
    log.propagate = False

    return log


log = setup_logger()
