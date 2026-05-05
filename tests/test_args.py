import pytest

from fuse.args import create_parser


class TestCreateParser:
    def setup_method(self):
        self.parser = create_parser()

    def test_expression_parsed(self):
        args = self.parser.parse_args(["[abc]"])
        assert args.expression == "[abc]"

    def test_no_expression_defaults_none(self):
        args = self.parser.parse_args([])
        assert args.expression is None

    def test_output_option(self):
        args = self.parser.parse_args(["-o", "out.txt", "[ab]"])
        assert args.output == "out.txt"

    def test_output_long(self):
        args = self.parser.parse_args(["--output", "out.txt", "[ab]"])
        assert args.output == "out.txt"

    def test_quiet_flag(self):
        args = self.parser.parse_args(["-q", "[ab]"])
        assert args.quiet is True

    def test_quiet_default(self):
        args = self.parser.parse_args(["[ab]"])
        assert args.quiet is False

    def test_non_interactive_flag(self):
        args = self.parser.parse_args(["-n", "[ab]"])
        assert args.non_interactive is True

    def test_non_interactive_default(self):
        args = self.parser.parse_args(["[ab]"])
        assert args.non_interactive is False

    def test_delimiter_option(self):
        args = self.parser.parse_args(["-d", ",", "[ab]"])
        assert args.delimiter == ","

    def test_delimiter_default(self):
        args = self.parser.parse_args(["[ab]"])
        assert args.delimiter == "\n"

    def test_buffer_option(self):
        args = self.parser.parse_args(["-b", "4KB", "[ab]"])
        assert args.buffer == "4KB"

    def test_buffer_default(self):
        args = self.parser.parse_args(["[ab]"])
        assert args.buffer == -1

    def test_workers_option(self):
        args = self.parser.parse_args(["-w", "4", "[ab]"])
        assert args.workers == 4

    def test_workers_default(self):
        args = self.parser.parse_args(["[ab]"])
        assert args.workers == 1

    def test_filter_option(self):
        args = self.parser.parse_args(["-F", "^a", "[ab]"])
        assert args.filter == "^a"

    def test_compresslevel_option(self):
        args = self.parser.parse_args(["-l", "5", "[ab]"])
        assert args.compresslevel == 5

    def test_start_option(self):
        args = self.parser.parse_args(["-S", "abc", "[ab]"])
        assert args.start == "abc"

    def test_end_option(self):
        args = self.parser.parse_args(["-E", "xyz", "[ab]"])
        assert args.end == "xyz"

    def test_file_option(self):
        args = self.parser.parse_args(["-f", "exprs.fuse"])
        assert args.expr_file == "exprs.fuse"

    def test_files_positional(self):
        args = self.parser.parse_args(["^", "file1.txt", "file2.txt"])
        assert args.files == ["file1.txt", "file2.txt"]

    def test_custom_prog(self):
        parser = create_parser(prog="custom")
        assert parser.prog == "custom"
