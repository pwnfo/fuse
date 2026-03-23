import re

from itertools import product
from typing import Generator, Any

from fuse.utils.classes import pattern_repl
from fuse.utils.files import secure_open


class ExprError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__("expression error: " + message)


class Node:
    __slots__ = ("base", "min_rep", "max_rep", "_sum_len", "_cached_cardinality")

    def __init__(
        self, base: str | list[str], min_rep: int = 1, max_rep: int = 1
    ) -> None:
        self.base = base if isinstance(base, list) else [base]
        self.min_rep = min_rep
        self.max_rep = max_rep
        self._sum_len: int | None = None
        self._cached_cardinality: int | None = None

    def __repr__(self) -> str:
        return f"<Node base={self.base!r} {{{self.min_rep},{self.max_rep}}}>"

    @property
    def cardinality(self) -> int:
        """calculates the total number of combinations this node generates."""
        if self._cached_cardinality is not None:
            return self._cached_cardinality

        count = 0
        base_len = len(self.base)
        for r in range(self.min_rep, self.max_rep + 1):
            if r == 0:
                count += 1
            else:
                count += base_len**r

        self._cached_cardinality = count
        return count

    def expand(self) -> Generator[str, None, None]:
        """standard generation using itertools."""
        min_r = self.min_rep
        max_r = self.max_rep
        base = self.base
        if min_r == 0 and max_r == 0:
            yield ""
            return
        for k in range(min_r, max_r + 1):
            if k == 0:
                yield ""
            else:
                for tup in product(base, repeat=k):
                    yield "".join(tup)

    def expand_resume(
        self, start_from: str
    ) -> Generator[tuple[str, str | None, bool], None, None]:
        """generates items starting from 'start_from' using seeking logic."""
        min_r = self.min_rep
        max_r = self.max_rep
        base = self.base

        if not start_from:
            for res in self.expand():
                yield res, None, True
            return

        for k in range(min_r, max_r + 1):
            if k == 0:
                yield "", start_from, False
                continue

            yield from self._product_resume_recursive(base, k, start_from)

    def _product_resume_recursive(
        self,
        pool: list[str],
        depth: int,
        target: str,
        current_prefix: str = "",
        seeking: bool = True,
    ) -> Generator[tuple[str, str | None, bool], None, None]:
        """recursive helper for resume generation (seeking/draining)."""
        if depth == 0:
            if seeking:
                if target.startswith(current_prefix):
                    remainder = target[len(current_prefix) :]
                    yield current_prefix, remainder, False
                elif current_prefix > target:
                    yield current_prefix, None, True
            else:
                yield current_prefix, None, True
            return

        if not seeking:
            remaining_depth = depth
            for tup in product(pool, repeat=remaining_depth):
                suffix = "".join(tup)
                yield current_prefix + suffix, None, True
            return

        found_path_in_this_level = False

        for item in pool:
            if found_path_in_this_level:
                yield from self._product_resume_recursive(
                    pool, depth - 1, target, current_prefix + item, seeking=False
                )
                continue

            candidate = current_prefix + item

            if target.startswith(candidate):
                found_path_in_this_level = True
                yield from self._product_resume_recursive(
                    pool, depth - 1, target, candidate, seeking=True
                )

            elif candidate.startswith(target):
                found_path_in_this_level = True
                yield from self._product_resume_recursive(
                    pool, depth - 1, target, candidate, seeking=False
                )

            else:
                pass

    def get_skipped_count(self, target: str) -> tuple[int, str | None]:
        """calculates how many items this node skips to reach a prefix of 'target'."""
        skipped_total = 0
        base_len = len(self.base)

        for k in range(self.min_rep, self.max_rep + 1):
            if k == 0:
                if target:
                    skipped_total += 1
                    continue
                else:
                    return skipped_total, ""

            res_skipped, res_remainder = self._calc_skip_recursive(self.base, k, target)

            if res_remainder is not None:
                return skipped_total + res_skipped, res_remainder

            skipped_total += base_len**k

        return skipped_total, None

    def _calc_skip_recursive(
        self, pool: list[str], depth: int, target: str
    ) -> tuple[int, str | None]:
        """recursive helper to calculate skip count (ranking logic)."""
        if depth == 0:
            return 0, target

        skipped = 0
        pool_len = len(pool)

        for i, item in enumerate(pool):
            if target.startswith(item):
                rem_target = target[len(item) :]
                rec_skipped, rec_remainder = self._calc_skip_recursive(
                    pool, depth - 1, rem_target
                )

                if rec_remainder is not None:
                    return skipped + rec_skipped, rec_remainder
                else:
                    skipped += pool_len ** (depth - 1)

            elif item.startswith(target):
                return skipped, ""

            else:
                skipped += pool_len ** (depth - 1)

        return skipped, None

    def get_item_at(self, index: int) -> str:
        """retrieves the item at a specific index."""
        base = self.base
        base_len = len(base)

        for r in range(self.min_rep, self.max_rep + 1):
            if r == 0:
                count = 1
            else:
                count = base_len**r

            if index < count:
                if r == 0:
                    return ""

                indices: list[int] = []
                temp = index
                for _ in range(r):
                    indices.append(temp % base_len)
                    temp //= base_len

                chars = [base[i] for i in reversed(indices)]
                return "".join(chars)

            index -= count

        raise IndexError("index out of range")


class FileNode(Node):
    __slots__ = ("_cached_lines", "_cached_sum_len")

    def __init__(self, files: list[str], min_rep: int = 1, max_rep: int = 1) -> None:
        super().__init__(files, min_rep, max_rep)
        self._cached_lines: list[str] | None = None
        self._cached_sum_len: int | None = None

    def __repr__(self) -> str:
        return f"<FileNode files={self.base!r} {{{self.min_rep},{self.max_rep}}}>"

    @property
    def lines(self) -> list[str]:
        """loads and caches lines from file paths."""
        cached = self._cached_lines
        if cached is not None:
            return cached
        out: list[str] = []
        for path in self.base:
            try:
                with secure_open(path, "r", encoding="utf-8", errors="ignore") as fp:
                    if not fp:
                        raise IOError
                    out.extend(ln.rstrip("\n\r") for ln in fp)
            except (IOError, OSError):
                raise ExprError(f"failed to open or read file: {path}")
        if not out:
            raise ExprError(f"file node produced no lines: {self.base}")
        self._cached_lines = out
        return out

    @property
    def cardinality(self) -> int:
        """returns total combinations based on file lines."""
        if self._cached_cardinality is not None:
            return self._cached_cardinality
        count = 0
        base_len = len(self.lines)
        for r in range(self.min_rep, self.max_rep + 1):
            if r == 0:
                count += 1
            else:
                count += base_len**r
        self._cached_cardinality = count
        return count

    def expand(self) -> Generator[str, None, None]:
        """standard file-based generation."""
        choices = self.lines
        min_r = self.min_rep
        max_r = self.max_rep
        if min_r == 0 and max_r == 0:
            yield ""
            return
        join = "".join
        for r in range(min_r, max_r + 1):
            if r == 0:
                yield ""
            else:
                for tup in product(choices, repeat=r):
                    yield join(tup)

    def expand_resume(
        self, start_from: str
    ) -> Generator[tuple[str, str | None, bool], None, None]:
        """resume generation for file content."""
        min_r = self.min_rep
        max_r = self.max_rep
        choices = self.lines

        if not start_from:
            for res in self.expand():
                yield res, None, True
            return

        for k in range(min_r, max_r + 1):
            if k == 0:
                yield "", start_from, False
                continue
            yield from self._product_resume_recursive(choices, k, start_from)

    def stats_info(self) -> tuple[int, int]:
        """calculates line count and total byte length for stats."""
        data = self.lines
        cached = self._cached_sum_len
        if cached is not None:
            return len(data), cached
        total_len = 0
        for line in data:
            total_len += len(line.encode("utf-8"))
        self._cached_sum_len = total_len
        return len(data), total_len

    def get_skipped_count(self, target: str) -> tuple[int, str | None]:
        """calculates skipped count using file lines as base."""
        skipped_total = 0
        choices = self.lines
        base_len = len(choices)

        for k in range(self.min_rep, self.max_rep + 1):
            if k == 0:
                if target:
                    skipped_total += 1
                    continue
                else:
                    return skipped_total, ""

            res_skipped, res_remainder = self._calc_skip_recursive(choices, k, target)
            if res_remainder is not None:
                return skipped_total + res_skipped, res_remainder

            skipped_total += base_len**k

        return skipped_total, None

    def get_item_at(self, index: int) -> str:
        """retrieves the item at a specific index."""
        base = self.lines
        base_len = len(base)

        for r in range(self.min_rep, self.max_rep + 1):
            if r == 0:
                count = 1
            else:
                count = base_len**r

            if index < count:
                if r == 0:
                    return ""

                indices: list[int] = []
                temp = index
                for _ in range(r):
                    indices.append(temp % base_len)
                    temp //= base_len

                chars = [base[i] for i in reversed(indices)]
                return "".join(chars)

            index -= count

        raise IndexError("index out of range")


class WordlistGenerator:
    BRACES_RE = re.compile(r"\{(\d+)(?:\s*,\s*(\d+))?\}")
    RANGE_RE = re.compile(r"\s*([0-9]+)\s*-\s*([0-9]+)\s*(?::\s*([+-]?\d+)\s*)?$")

    def _find_closing(self, s: str, start: int, closer: str) -> int:
        i = start
        n = len(s)
        while i < n:
            ch = s[i]
            if ch == "\\":
                i += 2
                continue
            if ch == closer:
                return i
            i += 1
        return -1

    def _parse_range(self, pattern: str, start_idx: int) -> tuple[list[str], int]:
        end_pos = self._find_closing(pattern, start_idx, "]")
        if end_pos == -1:
            raise ExprError("unclosed range: missing ']'.")
        inner = pattern[start_idx:end_pos]
        m = self.RANGE_RE.match(inner)
        if not m:
            raise ExprError("invalid range: expected '#[START-END[:STEP]]'.")
        r_start = int(m.group(1))
        r_end = int(m.group(2))
        step_str = m.group(3)
        step = int(step_str) if step_str else (1 if r_start <= r_end else -1)
        if step == 0:
            raise ExprError("invalid range: step cannot be zero.")
        if r_start < 0 or r_end < 0:
            raise ExprError("invalid range: start/end must be non-negative.")
        if (step > 0 and r_start > r_end) or (step < 0 and r_start < r_end):
            raise ExprError("invalid range sequence.")
        if step > 0:
            rng = range(r_start, r_end + 1, step)
        else:
            rng = range(r_start, r_end - 1, step)
        choices = [str(x) for x in rng]
        if not choices:
            raise ExprError("invalid range: produced no values.")
        return choices, end_pos + 1

    def _parse_class(
        self, pattern: str, start_idx: int, literal_mode: bool
    ) -> tuple[list[str], int]:
        closer = ")" if literal_mode else "]"
        end_pos = self._find_closing(pattern, start_idx, closer)
        if end_pos == -1:
            raise ExprError(f"unclosed class: missing '{closer}'.")
        inner = pattern[start_idx:end_pos]
        if not inner:
            raise ExprError("empty class is not allowed.")
        if literal_mode:
            return [inner], end_pos + 1
        if "|" not in inner and "\\" not in inner:
            return list(inner), end_pos + 1
        segments: list[str] = []
        buf: list[str] = []
        escape = False
        for ch in inner:
            if escape:
                buf.append(ch)
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == "|":
                segments.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        segments.append("".join(buf))
        choices = [s.strip() for s in segments if s.strip()]
        if not choices:
            raise ExprError("invalid character class contents.")
        return choices, end_pos + 1

    def tokenize(self, pattern: str) -> list[tuple[str, Any]]:
        pattern = pattern_repl(pattern)
        i = 0
        n = len(pattern)
        tokens: list[tuple[str, Any]] = []
        pr = pattern
        BR = self.BRACES_RE
        while i < n:
            c = pr[i]
            if c == "\\":
                if i + 1 >= n:
                    raise ExprError("invalid escape: ends with backslash.")
                tokens.append(("LIT", pr[i + 1]))
                i += 2
                continue
            if c == "#":
                if i + 1 < n and pr[i + 1] == "[":
                    choices, new_i = self._parse_range(pr, i + 2)
                    tokens.append(("RANGE", choices))
                    i = new_i
                else:
                    tokens.append(("LIT", "#"))
                    i += 1
                continue
            if c == "(":
                choices, new_i = self._parse_class(pr, i + 1, literal_mode=True)
                tokens.append(("CLASS", choices))
                i = new_i
                continue
            if c == "[":
                choices, new_i = self._parse_class(pr, i + 1, literal_mode=False)
                tokens.append(("CLASS", choices))
                i = new_i
                continue
            if c == "?":
                tokens.append(("QMARK", None))
                i += 1
                continue
            if c == "@":
                tokens.append(("FILE", None))
                i += 1
                continue
            if c == "{":
                m = BR.match(pr[i:])
                if m:
                    a = int(m.group(1))
                    b = int(m.group(2)) if m.group(2) is not None else a
                    if a > b:
                        raise ExprError("invalid repetition: min > max.")
                    tokens.append(("BRACES", (a, b)))
                    i += m.end()
                    continue
                else:
                    raise ExprError("invalid repetition syntax.")
            tokens.append(("LIT", c))
            i += 1
        return tokens

    def parse(
        self, tokens: list[tuple[str, Any]], files: list[str] | None = None
    ) -> list[Node | FileNode]:
        nodes: list[Node | FileNode] = []
        count_ft = 0
        for t, _ in tokens:
            if t == "FILE":
                count_ft += 1
        if count_ft:
            if not files:
                raise ExprError("pattern requires files but none provided.")
            if len(files) < 1:
                raise ExprError("files list is empty.")
            if count_ft == 1:
                file_groups = [files]
            else:
                if len(files) < count_ft:
                    raise ExprError(
                        f"pattern requires {count_ft} files, {len(files)} provided."
                    )
                file_groups = [[f] for f in files[:count_ft]]
        else:
            file_groups = []
        file_idx = 0
        i = 0
        length = len(tokens)
        while i < length:
            kind, val = tokens[i]
            min_rep = 1
            max_rep = 1
            if i + 1 < length:
                next_k, next_v = tokens[i + 1]
                if next_k == "QMARK":
                    min_rep, max_rep = 0, 1
                    i += 1
                elif next_k == "BRACES":
                    min_rep, max_rep = next_v
                    i += 1
            if kind == "LIT" or kind == "CLASS" or kind == "RANGE":
                nodes.append(Node(val, min_rep, max_rep))
            elif kind == "FILE":
                if file_idx >= len(file_groups):
                    raise ExprError("insufficient file assignments.")
                nodes.append(FileNode(file_groups[file_idx], min_rep, max_rep))
                file_idx += 1
            else:
                raise ExprError(f"unexpected token: {kind}")
            i += 1
        return nodes

    def _combine_resume(
        self, nodes: list[Node], idx: int, start_from: str | None
    ) -> Generator[str, None, None]:
        """recursive word combination generator with resume logic."""
        ln = len(nodes)
        if idx >= ln:
            if not start_from:
                yield ""
            return
        cur = nodes[idx]
        if start_from is None:
            for part in cur.expand():
                for suffix in self._combine_resume(nodes, idx + 1, None):
                    yield part + suffix
            return

        for part, remainder, is_full_mode in cur.expand_resume(start_from):
            next_target = None if is_full_mode else remainder
            for suffix in self._combine_resume(nodes, idx + 1, next_target):
                yield part + suffix

    def generate(
        self,
        nodes: list[Node | FileNode],
        start_from: str | None = None,
        end: str | None = None,
    ) -> Generator[str, None, None]:
        """starts the wordlist generation, optionally bounded by start_from and end."""
        iterator = self._combine_resume(nodes, 0, start_from)
        found = False if start_from else True

        for item in iterator:
            if not found:
                if item == start_from:
                    found = True
                continue

            yield item

            if end and item == end:
                break

    def _get_suffix_capacity(self, nodes: list[Node | FileNode], start_idx: int) -> int:
        """calculates total combinations of subsequent nodes."""
        total = 1
        for i in range(start_idx, len(nodes)):
            total *= nodes[i].cardinality
        return total

    def _calculate_skipped_count(
        self, nodes: list[Node | FileNode], target: str
    ) -> int:
        """calculates how many words exist strictly before 'target'."""
        skipped_count = 0
        current_target = target

        for i, node in enumerate(nodes):
            node_skipped, remainder = node.get_skipped_count(current_target)
            suffix_capacity = self._get_suffix_capacity(nodes, i + 1)
            skipped_count += node_skipped * suffix_capacity

            if remainder is None:
                break

            current_target = remainder
            if not current_target and i < len(nodes) - 1:
                break

        return skipped_count

    def get_word_at_index(self, nodes: list[Node | FileNode], index: int) -> str:
        """retrieves the word at a specific index."""
        result: list[str] = []
        for i, node in enumerate(nodes):
            suffix_cap = self._get_suffix_capacity(nodes, i + 1)
            node_idx = index // suffix_cap
            index %= suffix_cap
            result.append(node.get_item_at(node_idx))
        return "".join(result)

    def stats(
        self,
        nodes: list[Node | FileNode],
        sep_len: int = 1,
        start_from: str | None = None,
        end: str | None = None,
    ) -> tuple[int, int]:
        """calculates wordlist stats, adjusted for start_from and end range."""
        total_count = 1
        total_bytes = 0

        # calculate absolute total
        for node in nodes:
            if isinstance(node, FileNode):
                k, sum_len = node.stats_info()
            else:
                choices = node.base
                k = len(choices)
                cached = node._sum_len
                if cached is None:
                    s = 0
                    for s_item in choices:
                        s += len(str(s_item).encode("utf-8"))
                    node._sum_len = s
                    sum_len = s
                else:
                    sum_len = cached
            node_count = 0
            node_bytes = 0
            min_r = node.min_rep
            max_r = node.max_rep
            if min_r == 0 and max_r == 0:
                node_count = 1
                node_bytes = 0
            else:
                for r in range(min_r, max_r + 1):
                    if r == 0:
                        term_count = 1
                        term_bytes = 0
                    else:
                        term_count = k**r
                        term_bytes = r * (k ** (r - 1)) * sum_len
                    node_count += term_count
                    node_bytes += term_bytes
            total_count, total_bytes = total_count * node_count, (
                total_bytes * node_count
            ) + (node_bytes * total_count)

        full_total_bytes = int(total_bytes + (sep_len * total_count))
        full_total_count = int(total_count)

        if not start_from and not end:
            return full_total_bytes, full_total_count

        start_deduction = 0
        if start_from:
            start_idx = self._calculate_skipped_count(nodes, start_from)
            start_deduction = start_idx + 1

        end_cap = full_total_count
        if end:
            end_idx = self._calculate_skipped_count(nodes, end)
            end_cap = end_idx + 1

        # count represents range between start and end
        actual_count = end_cap - start_deduction

        if actual_count < 0:
            actual_count = 0

        # estimate bytes using proportional ratio
        if full_total_count > 0:
            ratio = actual_count / full_total_count
            actual_bytes = int(full_total_bytes * ratio)
        else:
            actual_bytes = 0

        return actual_bytes, actual_count
