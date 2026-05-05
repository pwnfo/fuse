import sys
import time

from threading import Event
from time import sleep
from typing import Any
from collections import deque

from fuse.core.formatters import format_size

# -------------------------------------
# progress bar configuration constants.
# -------------------------------------
_PROGRESS_EMA_ALPHA = 0.2  # exponential moving average smoothing factor.
_PROGRESS_SAMPLE_HISTORY = 4  # number of time samples to keep.
_PROGRESS_UPDATE_INTERVAL = 0.5  # seconds between progress updates.
_PROGRESS_RATE_BLEND_THRESHOLD = 1.0  # seconds before using instant rate.
_PROGRESS_RATE_BLEND_FACTOR = 0.5  # weight for blending rates.


def calc_rate(samples: deque, total_bytes: int, elapsed: float) -> str:
    """Calculate and format the current transfer rate"""
    if elapsed > 0:
        avg_rate = total_bytes / elapsed
    else:
        avg_rate = 0

    if len(samples) < 2:
        return format_size(avg_rate, d=2) + "/s" if avg_rate > 0 else "0 B/s"

    t0, b0 = samples[0]
    t1, b1 = samples[-1]

    dt = t1 - t0
    db = b1 - b0

    if dt <= 0 or db <= 0:
        return format_size(avg_rate, d=2) + "/s" if avg_rate > 0 else "0 B/s"

    inst_rate = db / dt

    if dt < _PROGRESS_RATE_BLEND_THRESHOLD:
        rate_val = (inst_rate * _PROGRESS_RATE_BLEND_FACTOR) + (
            avg_rate * _PROGRESS_RATE_BLEND_FACTOR
        )
    else:
        rate_val = inst_rate

    return format_size(rate_val, d=2) + "/s"


def get_progress(e: Event, r: Any, total: int = 100) -> None:
    """Display a progress bar in the terminal

    Args:
        e (Event): Event to signal progress termination.
        r (Any): Variable controlled by the writer to update the bytes written.
        total (int, optional): Total number of bytes to be written (maximum value for `r.value`). Defaults to 100.
    """
    sys.stdout.write("\033[?25l")  # remove the terminal cursor

    samples: deque = deque(maxlen=_PROGRESS_SAMPLE_HISTORY)
    curr_bytes = 0
    ema_rate = 0.0
    start_time = time.time()

    while r.value < total:
        try:
            # stops the loop immediately if the event is set
            if e.is_set():
                break

            # freezes if `r.value` is not updated
            if curr_bytes == r.value:
                continue

            curr_time = time.time()
            curr_bytes = r.value

            samples.append((curr_time, curr_bytes))
            elapsed_time = curr_time - start_time

            inst_rate = 0.0
            if len(samples) >= 2:
                t0, b0 = samples[0]
                t1, b1 = samples[-1]
                dt = t1 - t0
                db = b1 - b0

                if dt > 0:
                    inst_rate = db / dt

            if inst_rate <= 0 and elapsed_time > 0:
                inst_rate = curr_bytes / elapsed_time

            ema_rate = (
                (_PROGRESS_EMA_ALPHA * inst_rate) + (1 - _PROGRESS_EMA_ALPHA) * ema_rate
                if ema_rate > 0
                else inst_rate
            )

            rate = calc_rate(samples, curr_bytes, elapsed_time)

            progress_pct = int((curr_bytes / total) * 100)

            remaining_time = (total - curr_bytes) / ema_rate if ema_rate > 0 else 0
            mins, secs = divmod(int(remaining_time), 60)

            message = (
                f"Generating {format_size(curr_bytes, d=2)} / {format_size(total, d=2)} "
                f"[{progress_pct}%] @ {rate} ETA {mins:02d}:{secs:02d}"
            )

            # clears the terminal line
            # before updating progress
            sys.stdout.write("\033[2K\r")
            sys.stdout.write(message)
            sys.stdout.flush()

            sleep(_PROGRESS_UPDATE_INTERVAL)

        except KeyboardInterrupt:
            break

    # clears the line and redisplays
    # the terminal cursor before exiting.
    sys.stdout.write("\033[?25h\033[2K\r")
    sys.stdout.flush()
