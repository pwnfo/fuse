import sys
import logging

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from rich.markup import render


class PlainRichHandler(RichHandler):
    """Disable Rich text highlighting"""

    def render_message(self, record: logging.LogRecord, message: str) -> Text:
        return render(message)

    def get_level_text(self, record: logging.LogRecord) -> Text:
        return Text(record.levelname)


class FuseRichHandler(PlainRichHandler):
    """Writes warnings and errors to `sys.stderr`"""

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno < logging.WARNING:
            self.console = Console(file=sys.stdout)
        else:
            self.console = Console(file=sys.stderr)

        super().emit(record)


class FuseFormatter(logging.Formatter):
    """Uses the 'Warning:' prefix"""

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.WARNING:
            return f"[bold yellow]\\[WARN][/bold yellow] {record.getMessage()}"
        if record.levelno == logging.ERROR:
            return f"[bold red]\\[ERROR][/bold red] {record.getMessage()}"
        return record.getMessage()


def setup_logger() -> logging.Logger:
    log = logging.getLogger(__name__)

    handler = FuseRichHandler(
        markup=True,
        rich_tracebacks=True,
        show_time=False,
        show_level=False,
        show_path=False,
        keywords=[],
    )

    handler.setFormatter(FuseFormatter())

    log.setLevel(logging.INFO)
    log.addHandler(handler)
    log.propagate = False

    return log


log = setup_logger()
