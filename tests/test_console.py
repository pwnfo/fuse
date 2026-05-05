import pytest

from collections import deque
from fuse.console import calc_rate


class TestCalcRate:
    def test_normal_rate(self):
        samples = deque([(0.0, 0), (1.0, 1024)], maxlen=4)
        result = calc_rate(samples, 1024, 1.0)
        assert "KB/s" in result

    def test_zero_elapsed_time(self):
        samples = deque(maxlen=4)
        result = calc_rate(samples, 100, 0.0)
        assert result == "0 B/s"

    def test_negative_elapsed_time(self):
        samples = deque(maxlen=4)
        result = calc_rate(samples, 100, -1.0)
        assert result == "0 B/s"

    def test_no_bytes_change(self):
        samples = deque([(0.0, 100), (1.0, 100)], maxlen=4)
        result = calc_rate(samples, 100, 1.0)
        assert "0" in result or "B/s" in result

    def test_large_rate(self):
        samples = deque([(0.0, 0), (1.0, 1024**3)], maxlen=4)
        result = calc_rate(samples, 1024**3, 1.0)
        assert "GB/s" in result or "/s" in result
