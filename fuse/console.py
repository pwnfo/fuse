import sys
import time

from threading import Event
from time import sleep
from typing import Any

from .logger import FUSE_NOCOLOR
from .utils.formatters import format_size

if not FUSE_NOCOLOR:
    from rich.console import Console
    from rich.progress import (
        Progress,
        BarColumn,
        TextColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
    )

    console = Console()


def calc_rate(prev_bytes: int, curr_bytes: int, delta_time: float) -> str:
    if delta_time <= 0:
        return "--"
    rate_bytes_per_sec = (curr_bytes - prev_bytes) / delta_time
    return format_size(rate_bytes_per_sec, d=2) + "/s"


def get_progress(e: Event, r: Any, total: int = 100) -> None:
    """Show progress bar"""
    if FUSE_NOCOLOR:
        sys.stdout.write("\033[?25l")
        while r.value < total:
            if e.is_set():
                break
            _ = int((r.value / total) * 100)
            sys.stdout.write(f"{_}% Generating...\r")
            sys.stdout.flush()
            sleep(1)
        sys.stdout.write("\033[?25h")
        sys.stdout.write(" " * 20 + "\r")
        sys.stdout.flush()
        return
    last_bytes = 0
    last_time = time.time()
    with Progress(
        TaskProgressColumn(),
        BarColumn(style="grey37", complete_style="bold green", bar_width=40),
        TextColumn("[cyan]{task.fields[rate]}[/cyan]"),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("Generating", total=total, rate="--")
        while r.value < total:
            if e.is_set():
                break
            now = time.time()
            rate = calc_rate(last_bytes, r.value, now - last_time)
            progress.update(task, completed=r.value, rate=rate)
            last_bytes = r.value
            last_time = now
            sleep(0.1)
        progress.update(task, completed=total, refresh=True)


def console_input(prompt: str, fprompt: str | None = None, **kwargs: Any) -> str:
    if fprompt is None:
        fprompt = prompt
    if FUSE_NOCOLOR:
        return input(prompt)
    return console.input(fprompt, **kwargs)
