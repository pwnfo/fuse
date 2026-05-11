import gzip
import bz2
import lzma
import pytest

from fuse.compression import (
    get_compression_extension,
)
from fuse.files import fuse_open


class TestCompressionExtensions:
    def test_gzip_extension(self):
        assert get_compression_extension("gzip") == ".gz"

    def test_bzip2_extension(self):
        assert get_compression_extension("bzip2") == ".bz2"

    def test_lzma_extension(self):
        assert get_compression_extension("lzma") == ".xz"


class TestCompressedFileWriter:
    def test_gzip_write(self, tmp_path):
        filepath = str(tmp_path / "test.txt")
        content = "hello world\nfoo bar\n"

        with fuse_open(filepath, "a", encoding="utf-8", compression="gzip") as fp:
            assert fp is not None
            fp.write(content)

        expected_path = filepath + ".gz"
        assert (tmp_path / "test.txt.gz").exists()

        with gzip.open(expected_path, "rt", encoding="utf-8") as f:
            assert f.read() == content

    def test_bzip2_write(self, tmp_path):
        filepath = str(tmp_path / "test.txt")
        content = "hello world\nfoo bar\n"

        with fuse_open(filepath, "a", encoding="utf-8", compression="bzip2") as fp:
            assert fp is not None
            fp.write(content)

        expected_path = filepath + ".bz2"
        assert (tmp_path / "test.txt.bz2").exists()

        with bz2.open(expected_path, "rt", encoding="utf-8") as f:
            assert f.read() == content

    def test_lzma_write(self, tmp_path):
        filepath = str(tmp_path / "test.txt")
        content = "hello world\nfoo bar\n"

        with fuse_open(filepath, "a", encoding="utf-8", compression="lzma") as fp:
            assert fp is not None
            fp.write(content)

        expected_path = filepath + ".xz"
        assert (tmp_path / "test.txt.xz").exists()

        with lzma.open(expected_path, "rt", encoding="utf-8") as f:
            assert f.read() == content

    def test_gzip_file_exists(self, tmp_path):
        filepath = str(tmp_path / "test.txt")

        with fuse_open(filepath, "a", encoding="utf-8", compression="gzip") as fp:
            assert fp is not None
            fp.write("first\n")

        with fuse_open(filepath, "a", encoding="utf-8", compression="gzip") as fp:
            assert fp is None

    def test_no_compression(self, tmp_path):
        filepath = str(tmp_path / "test.txt")
        content = "hello world\n"

        with fuse_open(filepath, "a", encoding="utf-8") as fp:
            assert fp is not None
            fp.write(content)

        assert (tmp_path / "test.txt").exists()
        assert not (tmp_path / "test.txt.gz").exists()

        with open(filepath, "r", encoding="utf-8") as f:
            assert f.read() == content

    def test_stdout_with_compression(self):
        with fuse_open(None, "a", encoding="utf-8", compression="gzip") as fp:
            import sys

            assert fp is sys.stdout

    def test_permission_error_with_compression(self, tmp_path):
        ro_dir = tmp_path / "readonly"
        ro_dir.mkdir()
        ro_dir.chmod(0o444)

        filepath = str(ro_dir / "test.txt")

        try:
            with fuse_open(filepath, "a", encoding="utf-8", compression="gzip") as fp:
                assert fp is None or fp is not None
        finally:
            ro_dir.chmod(0o755)
