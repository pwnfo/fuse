from __future__ import annotations

from threading import Event
from time import sleep
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from rich.theme import Theme

from fuse.core.formatters import format_size

# -------------------------------------
# progress bar configuration constants.
# -------------------------------------
_PROGRESS_UPDATE_INTERVAL = 0.15

_THEME = Theme(
    {
        "accent": "rgb(255,120,0)",
        "accent_dim": "dim rgb(255,120,0)",
        "accent2": "rgb(255,165,40)",
    }
)


class FuseETAColumn(ProgressColumn):
    """ETA in orange"""

    def render(self, task: Any) -> Text:
        remaining = task.time_remaining

        if remaining is None:
            return Text("--:--", style="accent2")

        mins, secs = divmod(int(remaining), 60)
        return Text(f"{mins:02d}:{secs:02d}", style="accent_dim")


class SpeedColumn(ProgressColumn):
    """Transfer speed column"""

    def render(self, task: Any) -> Text:
        speed = task.speed or 0
        return Text(f"{format_size(speed, d=2)}/s", style="accent2")


def get_progress(e: Event, r: Any, total: int = 100) -> None:
    """Display progress in the terminal."""
    console = Console(theme=_THEME)

    with Progress(
        SpinnerColumn(style="accent_dim", spinner_name="dots"),
        TextColumn("[accent2]{task.percentage:>3.0f}%[/]"),
        TextColumn("[accent]{task.fields[done]} / {task.fields[total_fmt]}[/]"),
        SpeedColumn(),
        FuseETAColumn(),
        console=console,
        transient=True,
        expand=False,
        refresh_per_second=6,
    ) as progress:
        task_id = progress.add_task(
            "",
            total=total,
            done=format_size(0, d=2),
            total_fmt=format_size(total, d=2),
        )

        try:
            while not e.is_set():
                current = min(int(getattr(r, "value", 0)), total)

                progress.update(
                    task_id,
                    completed=current,
                    done=format_size(current, d=2),
                )

                if current >= total:
                    break

                sleep(_PROGRESS_UPDATE_INTERVAL)

        except KeyboardInterrupt:
            e.set()

        finally:
            final_value = min(int(getattr(r, "value", 0)), total)
            progress.update(
                task_id,
                completed=final_value,
                done=format_size(final_value, d=2),
            )
