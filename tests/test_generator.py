import pytest

from pathlib import Path
from fuse.core.generator import (
    ExprError,
    Node,
    FileNode,
    WordlistGenerator,
    BindDefNode,
    BindRefNode,
)


class TestNode:
    def test_single_char_cardinality(self):
        node = Node("a")
        assert node.cardinality == 1

    def test_multi_char_cardinality(self):
        node = Node(["a", "b", "c"])
        assert node.cardinality == 3

    def test_repeated_cardinality(self):
        node = Node(["a", "b"], min_rep=1, max_rep=2)
        # rep=1 => 2, rep=2 => 4
        assert node.cardinality == 6

    def test_optional_cardinality(self):
        node = Node(["a", "b"], min_rep=0, max_rep=1)
        # rep=0 => 1 (empty), rep=1 => 2
        assert node.cardinality == 3

    def test_zero_zero_cardinality(self):
        node = Node("x", min_rep=0, max_rep=0)
        assert node.cardinality == 1

    def test_cardinality_cached(self):
        node = Node(["x", "y"], min_rep=1, max_rep=3)
        c1 = node.cardinality
        c2 = node.cardinality
        assert c1 == c2

    def test_expand_single(self):
        node = Node("x")
        assert list(node.expand()) == ["x"]

    def test_expand_multi(self):
        node = Node(["a", "b"])
        assert list(node.expand()) == ["a", "b"]

    def test_expand_repeated(self):
        node = Node(["0", "1"], min_rep=1, max_rep=2)
        result = list(node.expand())
        assert result == ["0", "1", "00", "01", "10", "11"]

    def test_expand_optional(self):
        node = Node(["a"], min_rep=0, max_rep=1)
        result = list(node.expand())
        assert result == ["", "a"]

    def test_expand_zero_zero(self):
        node = Node("x", min_rep=0, max_rep=0)
        assert list(node.expand()) == [""]

    def test_expand_rep3(self):
        node = Node(["a", "b"], min_rep=3, max_rep=3)
        result = list(node.expand())
        assert len(result) == 8
        assert "aaa" in result
        assert "bbb" in result
        assert "aba" in result

    def test_get_item_at_basic(self):
        node = Node(["a", "b", "c"])
        assert node.get_item_at(0) == "a"
        assert node.get_item_at(1) == "b"
        assert node.get_item_at(2) == "c"

    def test_get_item_at_repeated(self):
        node = Node(["0", "1"], min_rep=1, max_rep=2)
        items = [node.get_item_at(i) for i in range(6)]
        assert items == ["0", "1", "00", "01", "10", "11"]

    def test_get_item_at_out_of_range(self):
        node = Node(["a"])
        with pytest.raises(IndexError):
            node.get_item_at(999)

    def test_repr(self):
        node = Node(["a", "b"], min_rep=2, max_rep=3)
        assert "Node" in repr(node)
        assert "{2,3}" in repr(node)


class TestFileNode:
    def test_lines_loaded(self, wordlist_file):
        node = FileNode([str(wordlist_file)])
        lines = node.lines
        assert lines == ["apple", "banana", "cherry"]

    def test_cardinality_from_file(self, wordlist_file):
        node = FileNode([str(wordlist_file)])
        assert node.cardinality == 3

    def test_expand_from_file(self, wordlist_file):
        node = FileNode([str(wordlist_file)])
        result = list(node.expand())
        assert result == ["apple", "banana", "cherry"]

    def test_expand_repeated(self, wordlist_file):
        node = FileNode([str(wordlist_file)], min_rep=1, max_rep=2)
        result = list(node.expand())
        # rep=1: 3, rep=2: 9
        assert len(result) == 12

    def test_stats_info(self, wordlist_file):
        node = FileNode([str(wordlist_file)])
        count, total_len = node.stats_info()
        assert count == 3
        assert total_len == len("apple") + len("banana") + len("cherry")

    def test_missing_file_raises(self, tmp_path):
        node = FileNode([str(tmp_path / "nope.txt")])
        with pytest.raises(ExprError, match="failed to open"):
            _ = node.lines

    def test_empty_file_raises(self, empty_file):
        node = FileNode([str(empty_file)])
        with pytest.raises(ExprError, match="no lines"):
            _ = node.lines

    def test_repr(self, wordlist_file):
        node = FileNode([str(wordlist_file)], min_rep=1, max_rep=2)
        assert "FileNode" in repr(node)

    def test_multiple_files(self, wordlist_file, wordlist_file_2):
        node = FileNode([str(wordlist_file), str(wordlist_file_2)])
        lines = node.lines
        assert lines == ["apple", "banana", "cherry", "cat", "dog"]

    def test_get_item_at(self, wordlist_file):
        node = FileNode([str(wordlist_file)])
        assert node.get_item_at(0) == "apple"
        assert node.get_item_at(1) == "banana"
        assert node.get_item_at(2) == "cherry"

    def test_get_item_at_out_of_range(self, wordlist_file):
        node = FileNode([str(wordlist_file)])
        with pytest.raises(IndexError):
            node.get_item_at(999)


class TestTokenize:
    def setup_method(self):
        self.gen = WordlistGenerator()

    def test_plain_literal(self):
        tokens = self.gen.tokenize("abc")
        assert tokens == [("LIT", "a"), ("LIT", "b"), ("LIT", "c")]

    def test_escaped_char(self):
        tokens = self.gen.tokenize("\\[")
        assert tokens == [("LIT", "[")]

    def test_escape_at_end_raises(self):
        with pytest.raises(ExprError, match="invalid escape"):
            self.gen.tokenize("\\")

    def test_simple_class(self):
        tokens = self.gen.tokenize("[abc]")
        assert tokens == [("CLASS", ["a", "b", "c"])]

    def test_class_with_pipe(self):
        tokens = self.gen.tokenize("[cat|dog]")
        assert tokens == [("CLASS", ["cat", "dog"])]

    def test_unclosed_class_raises(self):
        with pytest.raises(ExprError, match="unclosed character class"):
            self.gen.tokenize("[abc")

    def test_empty_class_raises(self):
        with pytest.raises(ExprError, match="empty character class"):
            self.gen.tokenize("[]")

    def test_literal_group(self):
        tokens = self.gen.tokenize("(hello)")
        assert tokens == [("CLASS", ["hello"])]

    def test_unclosed_literal_group_raises(self):
        with pytest.raises(ExprError, match="unclosed character class"):
            self.gen.tokenize("(hello")

    def test_numeric_range(self):
        tokens = self.gen.tokenize("#[0-3]")
        assert tokens == [("RANGE", ["0", "1", "2", "3"])]

    def test_numeric_range_with_step(self):
        tokens = self.gen.tokenize("#[0-10:2]")
        assert tokens == [("RANGE", ["0", "2", "4", "6", "8", "10"])]

    def test_descending_range(self):
        tokens = self.gen.tokenize("#[5-3]")
        assert tokens == [("RANGE", ["5", "4", "3"])]

    def test_descending_range_with_step(self):
        tokens = self.gen.tokenize("#[10-0:-3]")
        assert tokens == [("RANGE", ["10", "7", "4", "1"])]

    def test_range_zero_step_raises(self):
        with pytest.raises(ExprError, match="step cannot be zero"):
            self.gen.tokenize("#[0-5:0]")

    def test_range_invalid_direction_raises(self):
        with pytest.raises(ExprError, match="invalid range sequence"):
            self.gen.tokenize("#[5-0:1]")

    def test_unclosed_range_raises(self):
        with pytest.raises(ExprError, match="unclosed range"):
            self.gen.tokenize("#[0-5")

    def test_invalid_range_format_raises(self):
        with pytest.raises(ExprError, match="invalid range"):
            self.gen.tokenize("#[abc]")

    def test_hash_without_bracket_is_literal(self):
        tokens = self.gen.tokenize("#")
        assert tokens == [("LIT", "#")]

    def test_qmark_token(self):
        tokens = self.gen.tokenize("[ab]?")
        assert ("QMARK", None) in tokens

    def test_file_token(self):
        tokens = self.gen.tokenize("^")
        assert tokens == [("FILE", None)]

    def test_multiple_file_tokens(self):
        tokens = self.gen.tokenize("^^")
        assert tokens == [("FILE", None), ("FILE", None)]

    def test_braces_exact(self):
        tokens = self.gen.tokenize("[ab]{3}")
        assert ("BRACES", (3, 3)) in tokens

    def test_braces_range(self):
        tokens = self.gen.tokenize("[ab]{2,4}")
        assert ("BRACES", (2, 4)) in tokens

    def test_braces_min_gt_max_raises(self):
        with pytest.raises(ExprError, match="min > max"):
            self.gen.tokenize("[ab]{5,2}")

    def test_braces_invalid_syntax_raises(self):
        with pytest.raises(ExprError, match="invalid repetition"):
            self.gen.tokenize("[ab]{}")


class TestParse:
    def setup_method(self):
        self.gen = WordlistGenerator()

    def test_literal_nodes(self):
        tokens = self.gen.tokenize("abc")
        nodes = self.gen.parse(tokens)
        assert len(nodes) == 3
        for n in nodes:
            assert isinstance(n, Node)

    def test_class_node(self):
        tokens = self.gen.tokenize("[ab]")
        nodes = self.gen.parse(tokens)
        assert len(nodes) == 1
        assert nodes[0].base == ["a", "b"]

    def test_braces_applied(self):
        tokens = self.gen.tokenize("[ab]{2,3}")
        nodes = self.gen.parse(tokens)
        assert len(nodes) == 1
        assert nodes[0].min_rep == 2
        assert nodes[0].max_rep == 3

    def test_optional_applied(self):
        tokens = self.gen.tokenize("[ab]?")
        nodes = self.gen.parse(tokens)
        assert len(nodes) == 1
        assert nodes[0].min_rep == 0
        assert nodes[0].max_rep == 1

    def test_file_node_single_placeholder(self, wordlist_file):
        tokens = self.gen.tokenize("^")
        nodes = self.gen.parse(tokens, files=[str(wordlist_file)])
        assert len(nodes) == 1
        assert isinstance(nodes[0], FileNode)

    def test_file_node_multiple_placeholders(self, wordlist_file, wordlist_file_2):
        tokens = self.gen.tokenize("^^")
        nodes = self.gen.parse(tokens, files=[str(wordlist_file), str(wordlist_file_2)])
        assert len(nodes) == 2
        assert all(isinstance(n, FileNode) for n in nodes)

    def test_file_placeholder_without_files_raises(self):
        tokens = self.gen.tokenize("^")
        with pytest.raises(ExprError, match="requires files"):
            self.gen.parse(tokens)

    def test_insufficient_files_raises(self, wordlist_file):
        tokens = self.gen.tokenize("^^")
        with pytest.raises(ExprError, match="requires 2 files"):
            self.gen.parse(tokens, files=[str(wordlist_file)])

    def test_range_node(self):
        tokens = self.gen.tokenize("#[0-2]")
        nodes = self.gen.parse(tokens)
        assert len(nodes) == 1
        assert nodes[0].base == ["0", "1", "2"]


class TestGenerate:
    def setup_method(self):
        self.gen = WordlistGenerator()

    def _gen(self, expression, files=None, **kwargs):
        tokens = self.gen.tokenize(expression)
        nodes = self.gen.parse(tokens, files=files)
        return list(self.gen.generate(nodes, **kwargs))

    def test_single_literal(self):
        assert self._gen("a") == ["a"]

    def test_multi_literal(self):
        assert self._gen("ab") == ["ab"]

    def test_class_generates_all(self):
        result = self._gen("[abc]")
        assert result == ["a", "b", "c"]

    def test_two_classes(self):
        result = self._gen("[ab][12]")
        assert result == ["a1", "a2", "b1", "b2"]

    def test_literal_plus_class(self):
        result = self._gen("x[ab]")
        assert result == ["xa", "xb"]

    def test_class_repeated(self):
        result = self._gen("[01]{2}")
        assert result == ["00", "01", "10", "11"]

    def test_optional_class(self):
        result = self._gen("[ab]?")
        assert result == ["", "a", "b"]

    def test_numeric_range_generation(self):
        result = self._gen("#[1-5]")
        assert result == ["1", "2", "3", "4", "5"]

    def test_literal_group(self):
        result = self._gen("(hello)")
        assert result == ["hello"]

    def test_literal_group_with_class(self):
        result = self._gen("(prefix)[ab]")
        assert result == ["prefixa", "prefixb"]

    def test_pipe_class(self):
        result = self._gen("[cat|dog|fish]")
        assert result == ["cat", "dog", "fish"]

    def test_file_generation(self, wordlist_file):
        result = self._gen("^", files=[str(wordlist_file)])
        assert result == ["apple", "banana", "cherry"]

    def test_file_with_literal(self, wordlist_file):
        result = self._gen("^(suffix)", files=[str(wordlist_file)])
        assert result == ["applesuffix", "bananasuffix", "cherrysuffix"]

    def test_digit_shortcut(self):
        result = self._gen("/d")
        assert result == list("0123456789")

    def test_start_from(self):
        result = self._gen("[abc][12]", start_from="b1")
        assert result == ["b2", "c1", "c2"]

    def test_end(self):
        result = self._gen("[abc][12]", end="b1")
        assert result == ["a1", "a2", "b1"]

    def test_start_from_and_end(self):
        result = self._gen("[abc][12]", start_from="a2", end="b2")
        assert result == ["b1", "b2"]

    def test_complex_mixed(self):
        result = self._gen("[ab]{1,2}")
        assert result == ["a", "b", "aa", "ab", "ba", "bb"]

    def test_class_rep_with_literal(self):
        result = self._gen("x[01]{2}y")
        assert result == ["x00y", "x01y", "x10y", "x11y"]


class TestStats:
    def setup_method(self):
        self.gen = WordlistGenerator()

    def _stats(self, expression, files=None, **kwargs):
        tokens = self.gen.tokenize(expression)
        nodes = self.gen.parse(tokens, files=files)
        return self.gen.stats(nodes, **kwargs)

    def test_single_literal(self):
        total_bytes, total_words = self._stats("a")
        assert total_words == 1

    def test_class_count(self):
        _, total_words = self._stats("[abc]")
        assert total_words == 3

    def test_two_classes_count(self):
        _, total_words = self._stats("[ab][12]")
        assert total_words == 4

    def test_repeated_count(self):
        _, total_words = self._stats("[ab]{2}")
        assert total_words == 4

    def test_optional_count(self):
        _, total_words = self._stats("[ab]?")
        assert total_words == 3

    def test_delimiter_len_affects_bytes(self):
        b1, _ = self._stats("[ab]", delimiter_len=1)
        b2, _ = self._stats("[ab]", delimiter_len=5)
        assert b2 > b1

    def test_stats_with_start_from(self):
        _, total_words = self._stats("[abc][12]", start_from="b1")
        assert total_words == 3  # b2, c1, c2

    def test_stats_with_end(self):
        _, total_words = self._stats("[abc][12]", end="b1")
        assert total_words == 3  # a1, a2, b1

    def test_file_stats(self, wordlist_file):
        _, total_words = self._stats("^", files=[str(wordlist_file)])
        assert total_words == 3


class TestGetWordAtIndex:
    def setup_method(self):
        self.gen = WordlistGenerator()

    def test_first_word(self):
        tokens = self.gen.tokenize("[ab][12]")
        nodes = self.gen.parse(tokens)
        assert self.gen.get_word_at_index(nodes, 0) == "a1"

    def test_last_word(self):
        tokens = self.gen.tokenize("[ab][12]")
        nodes = self.gen.parse(tokens)
        assert self.gen.get_word_at_index(nodes, 3) == "b2"

    def test_middle_word(self):
        tokens = self.gen.tokenize("[ab][12]")
        nodes = self.gen.parse(tokens)
        assert self.gen.get_word_at_index(nodes, 1) == "a2"

    def test_consistency_with_generate(self):
        tokens = self.gen.tokenize("[abc][xyz]")
        nodes = self.gen.parse(tokens)
        generated = list(self.gen.generate(nodes))
        for idx, word in enumerate(generated):
            assert self.gen.get_word_at_index(nodes, idx) == word


class TestCalculateSkippedCount:
    def setup_method(self):
        self.gen = WordlistGenerator()

    def test_first_word_skips_zero(self):
        tokens = self.gen.tokenize("[ab][12]")
        nodes = self.gen.parse(tokens)
        assert self.gen._calculate_skipped_count(nodes, "a1") == 0

    def test_second_word_skips_one(self):
        tokens = self.gen.tokenize("[ab][12]")
        nodes = self.gen.parse(tokens)
        assert self.gen._calculate_skipped_count(nodes, "a2") == 1

    def test_last_word(self):
        tokens = self.gen.tokenize("[ab][12]")
        nodes = self.gen.parse(tokens)
        assert self.gen._calculate_skipped_count(nodes, "b2") == 3


class TestBinding:
    def setup_method(self):
        self.gen = WordlistGenerator()

    def _gen(self, expression, **kwargs):
        tokens = self.gen.tokenize(expression)
        nodes = self.gen.parse(tokens)
        return list(self.gen.generate(nodes, **kwargs))

    def test_basic_digit_binding(self):
        result = self._gen("<@d=/d>-<@d>")
        assert result == [f"{i}-{i}" for i in range(10)]

    def test_binding_with_class(self):
        result = self._gen("<@x=[ab]>-<@x>")
        assert result == ["a-a", "b-b"]

    def test_binding_stats_not_cartesian(self):
        tokens = self.gen.tokenize("<@d=/d>-<@d>")
        nodes = self.gen.parse(tokens)
        _, words = self.gen.stats(nodes)
        assert words == 10

    def test_binding_with_surrounding_literals(self):
        result = self._gen("pre-<@d=[ab]>-<@d>-suf")
        assert result == ["pre-a-a-suf", "pre-b-b-suf"]

    def test_two_independent_bindings(self):
        result = self._gen("<@a=[01]><@b=[xy]><@a><@b>")
        assert result == ["0x0x", "0y0y", "1x1x", "1y1y"]

    def test_undefined_ref_in_generate_raises(self):
        tokens = self.gen.tokenize("<@x>")
        nodes = self.gen.parse(tokens)
        with pytest.raises(ExprError, match="undefined variable"):
            list(self.gen.generate(nodes))

    def test_undefined_ref_in_stats_raises(self):
        tokens = self.gen.tokenize("<@x>")
        nodes = self.gen.parse(tokens)
        with pytest.raises(ExprError, match="undefined variable"):
            self.gen.stats(nodes)

    def test_tokenize_bind_def_produces_correct_token(self):
        tokens = self.gen.tokenize("<@v=[abc]>")
        assert len(tokens) == 1
        kind, val = tokens[0]
        assert kind == "BIND_DEF"
        name, inner_tokens = val
        assert name == "v"

    def test_tokenize_bind_ref_produces_correct_token(self):
        tokens = self.gen._tokenize_raw("<@v>")
        assert tokens == [("BIND_REF", "v")]

    def test_parse_creates_bind_def_node(self):
        tokens = self.gen.tokenize("<@d=/d>")
        nodes = self.gen.parse(tokens)
        assert len(nodes) == 1
        assert isinstance(nodes[0], BindDefNode)
        assert nodes[0].name == "d"

    def test_parse_creates_bind_ref_node(self):
        tokens = self.gen.tokenize("<@d=/d><@d>")
        nodes = self.gen.parse(tokens)
        assert len(nodes) == 2
        assert isinstance(nodes[1], BindRefNode)
        assert nodes[1].name == "d"

    def test_unclosed_binding_raises(self):
        with pytest.raises(ExprError, match="unclosed binding"):
            self.gen.tokenize("<@d=/d")

    def test_invalid_binding_name_raises(self):
        with pytest.raises(ExprError, match="invalid binding name"):
            self.gen.tokenize("<@123=abc>")

    def test_literal_lt_without_at_is_literal(self):
        tokens = self.gen._tokenize_raw("<hello>")
        assert tokens[0] == ("LIT", "<")

    def test_bind_def_cardinality(self):
        tokens = self.gen.tokenize("<@d=/d>")
        nodes = self.gen.parse(tokens)
        assert nodes[0].cardinality == 10

    def test_reuse_with_numeric_range(self):
        result = self._gen("<@n=#[1-3]>/<@n>")
        assert result == ["1/1", "2/2", "3/3"]
