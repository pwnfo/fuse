import pytest

from fuse.utils.formatters import format_size, parse_size, format_time


class TestFormatSize:
    def test_bytes(self):
        assert format_size(500) == "500 B"

    def test_kilobytes(self):
        assert format_size(1024) == "1 KB"

    def test_megabytes(self):
        assert format_size(1024**2) == "1 MB"

    def test_gigabytes(self):
        assert format_size(1024**3) == "1 GB"

    def test_terabytes(self):
        assert format_size(1024**4) == "1 TB"

    def test_petabytes(self):
        assert format_size(1024**5) == "1 PB"

    def test_exabytes(self):
        assert format_size(1024**6) == "1 EB"

    def test_zero(self):
        assert format_size(0) == "0 B"

    def test_decimal_precision(self):
        result = format_size(1536, d=2)
        assert result == "1.50 KB"

    def test_fractional_bytes(self):
        assert format_size(512) == "512 B"

    def test_just_under_kb(self):
        assert format_size(1023) == "1023 B"


# ─── parse_size ──────────────────────────────────────────────────────────────


class TestParseSize:
    def test_bytes_explicit(self):
        assert parse_size("100B") == 100

    def test_bytes_no_unit(self):
        assert parse_size("100") == 100

    def test_kilobytes(self):
        assert parse_size("1KB") == 1024

    def test_megabytes(self):
        assert parse_size("1MB") == 1024**2

    def test_gigabytes(self):
        assert parse_size("1GB") == 1024**3

    def test_terabytes(self):
        assert parse_size("1TB") == 1024**4

    def test_petabytes(self):
        assert parse_size("1PB") == 1024**5

    def test_fractional(self):
        assert parse_size("1.5KB") == 1536

    def test_whitespace_tolerance(self):
        assert parse_size("  100 B  ") == 100

    def test_case_insensitive(self):
        assert parse_size("1kb") == 1024

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="invalid format"):
            parse_size("abc")

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="invalid format"):
            parse_size("100XB")


# ─── format_time ─────────────────────────────────────────────────────────────


class TestFormatTime:
    def test_seconds_only(self):
        assert format_time(30) == "30 seconds"

    def test_zero_seconds(self):
        assert format_time(0) == "0 seconds"

    def test_minutes_and_seconds(self):
        assert format_time(90) == "1 minutes, 30 seconds"

    def test_hours_minutes_seconds(self):
        assert format_time(3661) == "1 hours, 1 minutes, 1 seconds"

    def test_exact_hour(self):
        assert format_time(3600) == "1 hours, 0 minutes, 0 seconds"

    def test_fractional_truncated(self):
        assert format_time(59.9) == "59 seconds"
