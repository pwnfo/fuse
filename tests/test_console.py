import pytest

from fuse.console import calc_rate


class TestCalcRate:
    def test_normal_rate(self):
        result = calc_rate(0, 1024, 1.0)
        assert "1.00KB/s" == result

    def test_zero_delta_time(self):
        result = calc_rate(0, 100, 0.0)
        assert result == "--"

    def test_negative_delta_time(self):
        result = calc_rate(0, 100, -1.0)
        assert result == "--"

    def test_no_change(self):
        result = calc_rate(100, 100, 1.0)
        assert "0.00B/s" == result

    def test_large_rate(self):
        result = calc_rate(0, 1024**3, 1.0)
        assert "GB/s" in result
