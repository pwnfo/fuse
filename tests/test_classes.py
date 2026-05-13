import pytest

from fuse.utils.classes import CHAR_CLASSES, pattern_repl


class TestCharClasses:
    def test_digits(self):
        assert CHAR_CLASSES["d"] == "0123456789"

    def test_non_zero_digits(self):
        assert CHAR_CLASSES["D"] == "123456789"

    def test_hex_lower(self):
        assert CHAR_CLASSES["h"] == "0123456789abcdef"

    def test_hex_upper(self):
        assert CHAR_CLASSES["H"] == "0123456789ABCDEF"

    def test_alpha_lower(self):
        assert CHAR_CLASSES["a"] == "abcdefghijklmnopqrstuvwxyz"

    def test_alpha_upper(self):
        assert CHAR_CLASSES["A"] == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def test_space(self):
        assert CHAR_CLASSES["s"] == " "

    def test_octal(self):
        assert CHAR_CLASSES["o"] == "01234567"

    def test_punctuation(self):
        assert CHAR_CLASSES["p"] == "!@#$%^&*()-_+="

    def test_letters(self):
        assert (
            CHAR_CLASSES["l"] == "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )

    def test_newline(self):
        assert CHAR_CLASSES["b"] == "\n"


class TestPatternReplOutsideBrackets:
    """Shortcut expansion when the shortcut appears outside [] or ()."""

    def test_digit_shortcut(self):
        result = pattern_repl("/d")
        assert result == "[0123456789]"

    def test_alpha_lower_shortcut(self):
        result = pattern_repl("/a")
        assert result == "[abcdefghijklmnopqrstuvwxyz]"

    def test_alpha_upper_shortcut(self):
        result = pattern_repl("/A")
        assert result == "[ABCDEFGHIJKLMNOPQRSTUVWXYZ]"

    def test_hex_lower_shortcut(self):
        result = pattern_repl("/h")
        assert result == "[0123456789abcdef]"

    def test_multiple_shortcuts(self):
        result = pattern_repl("/d/a")
        assert result == "[0123456789][abcdefghijklmnopqrstuvwxyz]"

    def test_shortcut_with_literal(self):
        result = pattern_repl("abc/d")
        assert result == "abc[0123456789]"

    def test_unknown_shortcut_passthrough(self):
        result = pattern_repl("/z")
        assert result == "/z"

    def test_plain_string_unmodified(self):
        result = pattern_repl("hello")
        assert result == "hello"


class TestPatternReplInsideBrackets:
    """Shortcut expansion inside [...] (inline expansion)."""

    def test_digit_inside_bracket(self):
        result = pattern_repl("[/d]")
        assert result == "[0123456789]"

    def test_mixed_literal_and_shortcut(self):
        result = pattern_repl("[abc/d]")
        assert result == "[abc0123456789]"

    def test_multiple_shortcuts_inside_bracket(self):
        result = pattern_repl("[/d/a]")
        assert result == "[0123456789abcdefghijklmnopqrstuvwxyz]"

    def test_unknown_shortcut_inside_bracket_passthrough(self):
        result = pattern_repl("[/z]")
        assert result == "[/z]"


class TestPatternReplLiteralGroup:
    """Inside (...), shortcuts should NOT be expanded (literal mode)."""

    def test_literal_group_no_expansion(self):
        result = pattern_repl("(/d)")
        assert result == "(/d)"


class TestPatternReplEscape:
    """Backslash-escaped characters should be preserved."""

    def test_escaped_slash(self):
        result = pattern_repl("\\/d")
        assert result == "\\/d"

    def test_escaped_bracket(self):
        result = pattern_repl("\\[abc]")
        assert result == "\\[abc]"

    def test_escaped_inside_bracket(self):
        result = pattern_repl("[\\|abc]")
        assert result == "[\\|abc]"


class TestPatternReplCustomWildcard:
    """Using a custom wildcard character instead of '/'."""

    def test_custom_wildcard(self):
        result = pattern_repl("@d", wc="@")
        assert result == "[0123456789]"

    def test_custom_wildcard_inside_bracket(self):
        result = pattern_repl("[@d]", wc="@")
        assert result == "[0123456789]"

    def test_multi_char_wildcard(self):
        result = pattern_repl("::d", wc="::")
        assert result == "[0123456789]"
