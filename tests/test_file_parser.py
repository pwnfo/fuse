import pytest

from pathlib import Path
from fuse.file_parser import InvalidSyntaxError, process_expr_file


class TestProcessExprFile:
    def test_simple_expressions(self, fuse_expr_file):
        results = list(process_expr_file(str(fuse_expr_file)))
        assert len(results) == 2
        assert results[0] == ("[abc]{2}", [])
        assert results[1] == ("[12]{1}", [])

    def test_comment_lines_ignored(self, tmp_path):
        fp = tmp_path / "comment.fuse"
        fp.write_text("# this is a comment\n[ab]\n# another\n[12]\n")
        results = list(process_expr_file(str(fp)))
        assert len(results) == 2
        assert results[0][0] == "[ab]"
        assert results[1][0] == "[12]"

    def test_empty_lines_ignored(self, tmp_path):
        fp = tmp_path / "empty.fuse"
        fp.write_text("\n\n[ab]\n\n\n[12]\n\n")
        results = list(process_expr_file(str(fp)))
        assert len(results) == 2

    def test_alias_expansion(self, fuse_expr_file_with_alias):
        results = list(process_expr_file(str(fuse_expr_file_with_alias)))
        assert len(results) == 1
        expr, files = results[0]
        assert "[0123456789]" in expr

    def test_alias_invalid_raises(self, tmp_path):
        fp = tmp_path / "bad_alias.fuse"
        fp.write_text("%alias\n")
        with pytest.raises(InvalidSyntaxError, match="requires 2 arguments"):
            list(process_expr_file(str(fp)))

    def test_alias_name_with_dollar_raises(self, tmp_path):
        fp = tmp_path / "bad_alias2.fuse"
        fp.write_text("%alias $name value\n")
        with pytest.raises(InvalidSyntaxError, match="cannot contain"):
            list(process_expr_file(str(fp)))

    def test_alias_name_with_semicolon_raises(self, tmp_path):
        fp = tmp_path / "bad_alias3.fuse"
        fp.write_text("%alias na;me value\n")
        with pytest.raises(InvalidSyntaxError, match="cannot contain"):
            list(process_expr_file(str(fp)))

    def test_file_include(self, fuse_expr_file_with_file_ref, wordlist_file):
        results = list(process_expr_file(str(fuse_expr_file_with_file_ref)))
        assert len(results) == 1
        expr, files = results[0]
        assert len(files) == 1
        assert str(wordlist_file) in files[0]

    def test_file_keyword_without_args_raises(self, tmp_path):
        fp = tmp_path / "bad_file.fuse"
        fp.write_text("%file\n[ab]\n")
        with pytest.raises(InvalidSyntaxError, match="requires 1 argument"):
            list(process_expr_file(str(fp)))

    def test_file_cleared_after_expr(self, tmp_path, wordlist_file):
        fp = tmp_path / "multi.fuse"
        fp.write_text(f"%file {wordlist_file}\n^[ab]\n[12]\n")
        results = list(process_expr_file(str(fp)))
        assert len(results) == 2
        # first expression has file, second does not
        assert len(results[0][1]) == 1
        assert len(results[1][1]) == 0

    def test_nonexistent_file_returns_empty(self, tmp_path):
        fake = tmp_path / "nonexistent" / "ghost.fuse"
        results = list(process_expr_file(str(fake)))
        assert results == []

    def test_relative_file_path(self, tmp_path):
        words = tmp_path / "words.txt"
        words.write_text("hello\nworld\n")
        fp = tmp_path / "rel.fuse"
        fp.write_text(f"%file ./words.txt\n^{{2}}\n")
        results = list(process_expr_file(str(fp)))
        assert len(results) == 1
        _, files = results[0]
        assert len(files) == 1
        assert Path(files[0]).exists()
