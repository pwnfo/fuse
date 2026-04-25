import sys
import pytest

from fuse.core.files import secure_open


class TestSecureOpen:
    def test_none_returns_stdout(self):
        with secure_open(None) as fp:
            assert fp is sys.stdout

    def test_read_existing_file(self, wordlist_file):
        with secure_open(str(wordlist_file), "r", encoding="utf-8") as fp:
            assert fp is not None
            content = fp.read()
            assert "apple" in content

    def test_write_new_file(self, tmp_path):
        path = tmp_path / "new_file.txt"
        with secure_open(str(path), "w", encoding="utf-8") as fp:
            assert fp is not None
            fp.write("hello")
        assert path.read_text() == "hello"

    def test_file_not_found(self, tmp_path):
        fake = tmp_path / "nonexistent" / "file.txt"
        with secure_open(str(fake), "r", encoding="utf-8") as fp:
            assert fp is None

    def test_is_a_directory(self, tmp_path):
        with secure_open(str(tmp_path), "r", encoding="utf-8") as fp:
            assert fp is None
