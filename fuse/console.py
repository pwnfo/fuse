import sys
import time

from threading import Event
from time import sleep
from typing import Any

from fuse.utils.formatters import format_size


def calc_rate(prev_bytes: int, curr_bytes: int, delta_time: float) -> str:
    if delta_time <= 0:
        return "--"
    rate_bytes_per_sec = (curr_bytes - prev_bytes) / delta_time
    return format_size(rate_bytes_per_sec, d=2) + "/s"


def get_progress(e: Event, r: Any, total: int = 100) -> None:
    """Show progress bar"""
    dots = 0
    sys.stdout.write("\033[?25l")

    while not r.ready:
        if e.is_set():
            return

        ret = "." * ((dots % 3) + 1)
        message = f"IDLE :: Searching for '{r.word}'{ret}    \r"
        sys.stdout.write(message)
        dots += 1
        sleep(0.5)

    prev_bytes = r.value
    prev_time = time.time()

    while r.value < total:
        if e.is_set():
            break
        curr_bytes = r.value
        curr_time = time.time()

        _ = int((r.value / total) * 100)

        delta_time = curr_time - prev_time
        rate = calc_rate(prev_bytes, curr_bytes, delta_time)

        message = f"[{r.value/1024:.0f}/{total/1024:.0f}] KB :: {rate} :: {_}% Generating...\r"
        sys.stdout.write(message)
        sys.stdout.flush()

        sleep(0.5)

    sys.stdout.write("\033[?25h")
    sys.stdout.write(" " * len(message) + "\r")
    sys.stdout.flush()
