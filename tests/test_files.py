import sys
import pytest

from fuse.utils.files import fuse_open


class TestSecureOpen:
    def test_none_returns_stdout(self):
        with fuse_open(None) as fp:
            assert fp is sys.stdout

    def test_read_existing_file(self, wordlist_file):
        with fuse_open(str(wordlist_file), "r", encoding="utf-8") as fp:
            assert fp is not None
            content = fp.read()
            assert "apple" in content

    def test_write_new_file(self, tmp_path):
        path = tmp_path / "new_file.txt"
        with fuse_open(str(path), "w", encoding="utf-8") as fp:
            assert fp is not None
            fp.write("hello")
        assert path.read_text() == "hello"

    def test_file_not_found(self, tmp_path):
        fake = tmp_path / "nonexistent" / "file.txt"
        with fuse_open(str(fake), "r", encoding="utf-8") as fp:
            assert fp is None

    def test_is_a_directory(self, tmp_path):
        with fuse_open(str(tmp_path), "r", encoding="utf-8") as fp:
            assert fp is None

    def test_compression_parameter_default_none(self):
        """Test that compression defaults to None."""
        with fuse_open(None, compression=None) as fp:
            assert fp is sys.stdout
