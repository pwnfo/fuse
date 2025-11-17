import re
from itertools import product
from typing import Generator, Any, List

from fuse.utils.classes import pattern_repl
from fuse.utils.files import r_open


class ExprError(Exception):
    def __init__(self, message: str, pos: int | None = None):
        if pos is not None:
            message = f"char #{pos + 1}: {message}"
        super().__init__(message)


class Node:
    __slots__ = ("base", "min_rep", "max_rep", "_sum_len")

    def __init__(
        self, base: str | list[str], min_rep: int = 1, max_rep: int = 1
    ) -> None:
        self.base = base if isinstance(base, list) else [base]
        self.min_rep = min_rep
        self.max_rep = max_rep
        self._sum_len: int | None = None

    def __repr__(self) -> str:
        return f"<Node base={self.base!r} {{{self.min_rep},{self.max_rep}}}>"

    def expand(self) -> Generator[str, None, None]:
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
                prod = product(base, repeat=k)
                join = "".join
                for tup in prod:
                    yield join(tup)


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
        cached = self._cached_lines
        if cached is not None:
            return cached
        out: list[str] = []
        for path in self.base:
            try:
                with r_open(path, "r", encoding="utf-8", errors="ignore") as fp:
                    if not fp:
                        raise IOError
                    out.extend(ln.rstrip("\n\r") for ln in fp)
            except (IOError, OSError):
                raise ExprError(f"failed to open or read file: {path}")
        if not out:
            raise ExprError(f"file node produced no lines: {self.base}")
        self._cached_lines = out
        return out

    def stats_info(self) -> tuple[int, int]:
        data = self.lines
        cached = self._cached_sum_len
        if cached is not None:
            return len(data), cached
        total_len = 0
        for line in data:
            total_len += len(line.encode("utf-8"))
        self._cached_sum_len = total_len
        return len(data), total_len

    def expand(self) -> Generator[str, None, None]:
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
            raise ExprError("unclosed range: missing ']'.", start_idx - 2)
        inner = pattern[start_idx:end_pos]
        m = self.RANGE_RE.match(inner)
        if not m:
            raise ExprError(
                "invalid range: expected '#[START-END[:STEP]]'.", start_idx - 2
            )
        r_start = int(m.group(1))
        r_end = int(m.group(2))
        step_str = m.group(3)
        step = int(step_str) if step_str else (1 if r_start <= r_end else -1)
        if step == 0:
            raise ExprError("invalid range: STEP cannot be zero.", start_idx - 2)
        if r_start < 0 or r_end < 0:
            raise ExprError(
                "invalid range: START/END must be non-negative.", start_idx - 2
            )
        if (step > 0 and r_start > r_end) or (step < 0 and r_start < r_end):
            raise ExprError("invalid range sequence.", start_idx - 2)
        if step > 0:
            rng = range(r_start, r_end + 1, step)
        else:
            rng = range(r_start, r_end - 1, step)
        choices = [str(x) for x in rng]
        if not choices:
            raise ExprError("invalid range: produced no values.", start_idx - 2)
        return choices, end_pos + 1

    def _parse_class(
        self, pattern: str, start_idx: int, literal_mode: bool
    ) -> tuple[list[str], int]:
        closer = ")" if literal_mode else "]"
        end_pos = self._find_closing(pattern, start_idx, closer)
        if end_pos == -1:
            raise ExprError(f"unclosed class: missing '{closer}'.", start_idx - 1)
        inner = pattern[start_idx:end_pos]
        if not inner:
            raise ExprError("empty class is not allowed.", start_idx - 1)
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
            raise ExprError("invalid character class contents.", start_idx - 1)
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
                    raise ExprError("invalid escape: ends with backslash.", i)
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
                        raise ExprError("invalid repetition: MIN > MAX.", i)
                    tokens.append(("BRACES", (a, b)))
                    i += m.end()
                    continue
                else:
                    raise ExprError("invalid repetition syntax.", i)
            tokens.append(("LIT", c))
            i += 1
        return tokens

    def parse(
        self, tokens: list[tuple[str, Any]], files: List[str] | None = None
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
                raise ExprError(f"unexpected token: {kind}", i)
            i += 1
        return nodes

    def _combine(self, nodes: list[Node], idx: int) -> Generator[str, None, None]:
        ln = len(nodes)
        if idx >= ln:
            yield ""
            return
        cur = nodes[idx]
        for part in cur.expand():
            for suffix in self._combine(nodes, idx + 1):
                yield part + suffix

    def generate(
        self, nodes: list[Node | FileNode], start_from: str | None = None
    ) -> Generator[str, None, None]:
        iterator = self._combine(nodes, 0)
        if start_from:
            found = False
            for item in iterator:
                if found:
                    yield item
                elif item == start_from:
                    found = True
                    yield item
        else:
            yield from iterator

    def stats(self, nodes: list[Node | FileNode], sep_len: int = 1) -> tuple[int, int]:
        total_count = 1
        total_bytes = 0
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
        return int(total_bytes + (sep_len * total_count)), int(total_count)
