import pytest

from fuse.cli import format_expression, GenerateOptions


class TestFormatExpression:
    def test_no_files(self):
        expr, files = format_expression("[ab]^", [])
        assert expr == "[ab]^"
        assert files == []

    def test_inline_file(self):
        expr, files = format_expression("^[ab]", ["//inline_content"])
        assert expr == "inline_content[ab]"
        assert files == []

    def test_multiple_inline_files(self):
        expr, files = format_expression("^-^", ["//first", "//second"])
        assert expr == "first-second"
        assert files == []

    def test_regular_file_passthrough(self):
        expr, files = format_expression("^", ["words.txt"])
        assert expr == "^"
        assert files == ["words.txt"]

    def test_mixed_inline_and_regular(self):
        expr, files = format_expression("^[ab]^", ["//inline", "words.txt"])
        assert expr == "inline[ab]^"
        assert files == ["words.txt"]

    def test_inline_replaces_only_first_caret(self):
        expr, files = format_expression("^^", ["//first"])
        assert expr == "first^"
        assert files == []

    def test_escaped_caret_not_replaced(self):
        expr, files = format_expression("\\^", ["//nope"])
        assert expr == "\\^"
        assert files == []


class TestGenerateOptions:
    def test_defaults(self):
        opts = GenerateOptions(
            filename=None,
            buffering=-1,
            quiet_mode=False,
            delimiter="\n",
            wrange=(None, None),
            filter=None,
            threads=1,
            flush_limit=512 * 1024,
            compresslevel=None,
        )
        assert opts.filename is None
        assert opts.buffering == -1
        assert opts.quiet_mode is False
        assert opts.delimiter == "\n"
        assert opts.wrange == (None, None)
        assert opts.filter is None
        assert opts.threads == 1
        assert opts.flush_limit == 512 * 1024
        assert opts.compresslevel == None

    def test_with_values(self):
        opts = GenerateOptions(
            filename="out.txt",
            buffering=4096,
            quiet_mode=True,
            delimiter=",",
            wrange=("a", "z"),
            filter="^[a-z]+$",
            threads=4,
            flush_limit=256 * 1024,
            compresslevel=5,
        )
        assert opts.filename == "out.txt"
        assert opts.buffering == 4096
        assert opts.quiet_mode is True
        assert opts.delimiter == ","
        assert opts.wrange == ("a", "z")
        assert opts.filter == "^[a-z]+$"
        assert opts.threads == 4
        assert opts.flush_limit == 256 * 1024
        assert opts.compresslevel == 5
