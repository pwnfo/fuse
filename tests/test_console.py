import pytest
from unittest.mock import Mock, patch
from threading import Event

from rich.text import Text

from fuse.console import FuseETAColumn, SpeedColumn, get_progress


class DummyTask:
    def __init__(self, time_remaining=None, speed=None):
        self.time_remaining = time_remaining
        self.speed = speed


class TestFuseETAColumn:
    def test_render_none(self):
        col = FuseETAColumn()
        task = DummyTask(time_remaining=None)
        res = col.render(task)
        assert isinstance(res, Text)
        assert res.plain == "--:--"
        assert res.style == "accent2"

    def test_render_zero(self):
        col = FuseETAColumn()
        task = DummyTask(time_remaining=0)
        res = col.render(task)
        assert res.plain == "00:00"
        assert res.style == "accent_dim"

    def test_render_minutes_and_seconds(self):
        col = FuseETAColumn()
        task = DummyTask(time_remaining=65)
        res = col.render(task)
        assert res.plain == "01:05"
        assert res.style == "accent_dim"

    def test_render_hours(self):
        col = FuseETAColumn()
        task = DummyTask(time_remaining=3600)
        res = col.render(task)
        assert res.plain == "60:00"
        assert res.style == "accent_dim"


class TestSpeedColumn:
    def test_render_none(self):
        col = SpeedColumn()
        task = DummyTask(speed=None)
        res = col.render(task)
        assert isinstance(res, Text)
        assert "0.00 B/s" in res.plain
        assert res.style == "accent2"

    def test_render_value(self):
        col = SpeedColumn()
        task = DummyTask(speed=1024)
        res = col.render(task)
        assert "1.00 KB/s" in res.plain
        assert res.style == "accent2"


class DummyValue:
    def __init__(self, value=0):
        self.value = value


class TestGetProgress:
    @patch("fuse.console.sleep")
    def test_get_progress_completes(self, mock_sleep):
        e = Event()
        r = DummyValue(100)
        get_progress(e, r, total=100)
        assert mock_sleep.call_count == 0

    @patch("fuse.console.sleep")
    def test_get_progress_updates(self, mock_sleep):
        e = Event()
        r = DummyValue(0)

        def update_value(*args, **kwargs):
            r.value += 50

        mock_sleep.side_effect = update_value

        get_progress(e, r, total=100)
        assert mock_sleep.call_count > 0

    @patch("fuse.console.sleep")
    def test_get_progress_keyboard_interrupt(self, mock_sleep):
        e = Event()
        r = DummyValue(50)
        mock_sleep.side_effect = KeyboardInterrupt
        get_progress(e, r, total=100)
        assert e.is_set()
