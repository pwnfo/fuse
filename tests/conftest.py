import os
import pytest
import tempfile

from pathlib import Path


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Provides a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def wordlist_file(tmp_path: Path) -> Path:
    """Creates a temporary wordlist file with sample words."""
    filepath = tmp_path / "words.txt"
    filepath.write_text("apple\nbanana\ncherry\n", encoding="utf-8")
    return filepath


@pytest.fixture
def wordlist_file_2(tmp_path: Path) -> Path:
    """Creates a second temporary wordlist file."""
    filepath = tmp_path / "words2.txt"
    filepath.write_text("cat\ndog\n", encoding="utf-8")
    return filepath


@pytest.fixture
def empty_file(tmp_path: Path) -> Path:
    """Creates an empty temporary file."""
    filepath = tmp_path / "empty.txt"
    filepath.write_text("", encoding="utf-8")
    return filepath


@pytest.fixture
def fuse_expr_file(tmp_path: Path) -> Path:
    """Creates a fuse expression file (.fuse)."""
    filepath = tmp_path / "test.fuse"
    filepath.write_text("[abc]{2}\n[12]{1}\n", encoding="utf-8")
    return filepath


@pytest.fixture
def fuse_expr_file_with_define(tmp_path: Path) -> Path:
    """Creates a fuse expression file with aliases."""
    filepath = tmp_path / "define.fuse"
    filepath.write_text("%define DIGITS [0123456789]\n$DIGITS;{2}\n", encoding="utf-8")
    return filepath


@pytest.fixture
def fuse_expr_file_with_file_ref(tmp_path: Path, wordlist_file: Path) -> Path:
    """Creates a fuse expression file referencing an external file."""
    filepath = tmp_path / "fileref.fuse"
    filepath.write_text(
        f"%include {wordlist_file}\n^[ab]\n",
        encoding="utf-8",
    )
    return filepath


@pytest.fixture
def output_file(tmp_path: Path) -> Path:
    """Returns a path for an output file (does not create it)."""
    return tmp_path / "output.txt"
