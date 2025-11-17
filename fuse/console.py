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
    """Show progress bar with ETA"""
    message = ""
    start_time = time.time()

    sys.stdout.write("\033[?25l")

    prev_bytes = r.value
    prev_time = time.time()

    while r.value < total:
        if e.is_set():
            break
        curr_bytes = r.value
        curr_time = time.time()
        progress_pct = int((r.value / total) * 100)

        delta_time = curr_time - prev_time
        rate = calc_rate(prev_bytes, curr_bytes, delta_time)

        elapsed_time = curr_time - start_time
        avg_rate = r.value / elapsed_time if elapsed_time > 0 else 0
        remaining_time = (total - r.value) / avg_rate if avg_rate > 0 else 0
        mins, secs = divmod(int(remaining_time), 60)

        message = (
            f"Generating {format_size(r.value, d=2)} / {format_size(total, d=2)} "
            f"[{progress_pct}%] @ {rate} | ETA {mins:02d}:{secs:02d}    \r"
        )
        sys.stdout.write(message)
        sys.stdout.flush()

        prev_bytes = curr_bytes
        prev_time = curr_time
        sleep(0.5)

    sys.stdout.write("\033[?25h")
    sys.stdout.write(" " * len(message) + "\r")
    sys.stdout.flush()
