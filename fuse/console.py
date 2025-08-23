import os
import sys
import logging

from threading import Event
from time import sleep
from typing import Any

fusenv = os.environ.get("FUSE_CLI_NOCOLOR")
nocolor = fusenv is not None and fusenv != "0"

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.progress import (
        Progress,
        BarColumn,
        TextColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
    )

    class FuseRichHandler(RichHandler):
        fuse_console = Console(file=sys.stdout)
        fuse_err_console = Console(file=sys.stderr)

        def emit(self, record: logging.LogRecord) -> Any:

            # joga erros no stderr
            if record.levelno < logging.WARNING:
                self.console = self.fuse_console
            else:
                self.console = self.fuse_err_console
            super().emit(record)

    console = Console()

except ModuleNotFoundError:
    nocolor = True


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

    if nocolor:
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


def get_progress(e: Event, r: Any, total: int = 100) -> None:
    """Show progress bar to stdout"""
    if nocolor:
        while r.value < total:
            if e.is_set():
                break
            _ = int((r.value / total) * 100)
            sys.stdout.write(f"{_}% Generating...\r")
            sys.stdout.flush()
            sleep(0.5)
        sys.stdout.write(" " * 20 + "\r")
        sys.stdout.flush()
        return
    with Progress(
        TaskProgressColumn(),
        BarColumn(style="grey37", complete_style="bold green", bar_width=40),
        TextColumn(
            "[cyan]([bold]{task.completed}[/bold]/[bold]{task.total}[/bold])[/cyan]"
        ),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Generating", total=total)
        while r.value < total:
            if e.is_set():
                break
            progress.update(task, completed=r.value)
            sleep(0.1)
        progress.update(task, completed=total, refresh=True)


def console_input(prompt: str, fprompt: str | None = None, **kwargs: Any) -> str:
    if fprompt is None:
        fprompt = prompt
    if nocolor:
        return input(prompt)
    return console.input(fprompt, **kwargs)


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
