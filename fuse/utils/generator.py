import re
import sys

from itertools import product
from typing import Generator, Any, List, Never

from fuse.utils.classes import pattern_repl
from fuse.utils.files import r_open


class ExprError(Exception):
    def __init__(self, message: str, pos: int | None = None):
        if pos is not None:
            message = f"char #{pos + 1}: {message}"
        super().__init__(message)


class Node:
    def __init__(self, base: str | list, min_rep: int = 1, max_rep: int = 1) -> None:
        self.base = base
        self.min_rep = min_rep
        self.max_rep = max_rep

    def __repr__(self) -> str:
        return f"<Node base={self.base!r} {{{self.min_rep},{self.max_rep}}}>"

    def expand(self) -> Generator[str, None, None]:
        if isinstance(self.base, list):
            choices = self.base
        else:
            choices = [self.base]

        for k in range(self.min_rep, self.max_rep + 1):
            if k == 0:
                yield ""
            else:
                for tup in product(choices, repeat=k):
                    yield "".join(tup)


class FileNode(Node):
    def __repr__(self) -> str:
        return f"<FileNode files={self.base!r} {{{self.min_rep},{self.max_rep}}}>"

    def _collect_lines(self) -> list[str] | Never:
        if hasattr(self, "_lines"):
            self._lines: list[str]
            return self._lines

        lines: list[str] = []
        for path in self.base:
            with r_open(path, "r", encoding="utf-8", errors="ignore") as fp:
                if fp:
                    for ln in fp:
                        ln = ln.rstrip("\n\r")
                        lines.append(ln)
                else:
                    sys.exit(1)
        if not lines:
            raise ExprError("file node produced no lines (empty files?).")

        self._lines = lines
        return lines

    def stats_info(self) -> tuple[int, int] | Never:
        k = 0
        sum_len = 0
        for path in self.base:
            with r_open(path, "r", encoding="utf-8", errors="ignore") as fp:
                if fp:
                    for ln in fp:
                        ln = ln.rstrip("\n\r")
                        k += 1
                        sum_len += len(ln.encode("utf-8"))
                else:
                    sys.exit(1)
        if k == 0:
            raise ExprError("file node produced no lines (empty files?).")
        return k, sum_len

    def expand(self) -> Generator[str, None, None]:
        choices = self._collect_lines()
        k = len(choices)

        for r in range(self.min_rep, self.max_rep + 1):
            if r == 0:
                yield ""
            else:
                for tup in product(choices, repeat=r):
                    yield "".join(tup)


class Gen:
    BRACES_RE = re.compile(r"\{(\d+)(?:\s*,\s*(\d+))?\}")

    def tokenize(self, pattern: str) -> list[tuple[str, Any]]:
        pattern = pattern_repl(pattern)
        i, n = 0, len(pattern)
        tokens: list[tuple[str, Any]] = []

        while i < n:
            c = pattern[i]
            if c == "\\":
                if i + 1 < n:
                    tokens.append(("LIT", pattern[i + 1]))
                    i += 2
                else:
                    raise ExprError(
                        "invalid escape: pattern ends with a single backslash '\\'.", i
                    )
            elif c == "#":
                if i + 1 < n and pattern[i + 1] == "[":
                    match = re.search(r"(?<!\\)\]", pattern[i + 2 :])
                    if match is None:
                        raise ExprError("unclosed range: missing ']'.", i)
                    j = i + 2 + match.start()
                    inner = pattern[i + 2 : j]
                    m = re.match(
                        r"\s*([0-9]+)\s*-\s*([0-9]+)\s*(?::\s*([+-]?\d+)\s*)?$", inner
                    )
                    if not m:
                        raise ExprError(
                            "invalid range: expected '#[START-END[:STEP]]'.", i
                        )
                    start = int(m.group(1))
                    end = int(m.group(2))
                    step_str = m.group(3)
                    if step_str is None:
                        step = 1 if start <= end else -1
                    else:
                        step = int(step_str)
                    if step == 0:
                        raise ExprError("invalid range: STEP cannot be zero.", i)
                    if start < 0 or end < 0:
                        raise ExprError(
                            "invalid range: START and END must be non-negative integers.",
                            i,
                        )
                    if step > 0 and start > end:
                        raise ExprError(
                            "invalid range: START greater than END for positive STEP.",
                            i,
                        )
                    if step < 0 and start < end:
                        raise ExprError(
                            "invalid range: START less than END for negative STEP.", i
                        )
                    if step > 0:
                        rng = range(start, end + 1, step)
                    else:
                        rng = range(start, end - 1, step)
                    choices = [str(x) for x in rng]
                    if not choices:
                        raise ExprError("invalid range: produced no values.", i)
                    tokens.append(("RANGE", choices))
                    i = j + 1
                else:
                    tokens.append(("LIT", "#"))
                    i += 1
            elif c == "(":
                match = re.search(r"(?<!\\)\]", pattern[i + 1 :])
                if match is None:
                    raise ExprError("unclosed literal class: missing ')'.", i)
                j = i + 1 + match.start()
                inner = pattern[i + 1 : j]
                if inner == "":
                    raise ExprError("empty literal class '()' is not allowed.", i)
                tokens.append(("CLASS", [inner]))
                i = j + 1
            elif c == "[":
                match = re.search(r"(?<!\\)\]", pattern[i + 1 :])
                if match is None:
                    raise ExprError("unclosed character class: missing ']'.", i)
                j = i + 1 + match.start()
                inner = pattern[i + 1 : j]
                if inner == "":
                    raise ExprError("empty character class '[]' is not allowed.", i)
                if "|" in inner:
                    segments = []
                    buf = []
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

                    segments = [s.strip() for s in segments]
                    segments = [s for s in segments if s != ""]

                    if not segments:
                        raise ExprError(
                            "invalid character class contents inside [...].", i
                        )

                    if len(segments) > 1:
                        choices = segments
                    else:
                        choices = [segments[0]]

                    tokens.append(("CLASS", choices))
                    i = j + 1
                else:
                    tokens.append(("CLASS", list(inner)))
                    i = j + 1
            elif c == "?":
                tokens.append(("QMARK", None))
                i += 1
            elif c == "@":
                tokens.append(("FILE", None))
                i += 1
            elif c == "{":
                m = self.BRACES_RE.match(pattern[i:])
                if not m:
                    raise ExprError(
                        "invalid repetition syntax: expected '{R}' or '{MIN, MAX}'.", i
                    )
                a = int(m.group(1))
                b = m.group(2)
                if b is None:
                    tokens.append(("BRACES", (a, a)))
                else:
                    b_int = int(b)
                    if a > b_int:
                        raise ExprError(
                            "invalid repetition range: MIN cannot be greater than MAX in '{MIN, MAX}'.",
                            i,
                        )
                    tokens.append(("BRACES", (a, b_int)))
                i += m.end()
            else:
                tokens.append(("LIT", c))
                i += 1
        return tokens

    def parse(
        self, tokens: list[tuple[str, Any]], files: List[str] | None = None
    ) -> list[Node | FileNode]:
        i = 0
        L = len(tokens)
        nodes: list[Node] = []

        file_token_count = sum(1 for t, _ in tokens if t == "FILE")
        file_assignments: List[List[str]] = []

        if file_token_count == 0:
            file_assignments = []
        else:
            if files is None:
                raise ExprError(
                    "pattern contains '@' file placeholder but no files were provided.",
                    i,
                )
            if file_token_count == 1:
                file_assignments = [files]
            else:
                if len(files) < file_token_count:
                    raise ExprError(
                        f"pattern requires {file_token_count} file(s) (one per '@'), but only {len(files)} provided.",
                        i,
                    )
                file_assignments = [[f] for f in files[:file_token_count]]

        current_file_idx = 0

        while i < L:
            t, val = tokens[i]
            if t in ("LIT", "CLASS", "RANGE"):
                base = val
                min_rep, max_rep = 1, 1
                if i + 1 < L:
                    nt, nval = tokens[i + 1]
                    if nt == "QMARK":
                        min_rep, max_rep = 0, 1
                        i += 1
                    elif nt == "BRACES":
                        min_rep, max_rep = nval
                        i += 1
                nodes.append(Node(base, min_rep, max_rep))
            elif t == "FILE":
                if not file_assignments:
                    raise ExprError("no files assigned for '@' token", i)
                paths = file_assignments[current_file_idx]
                current_file_idx += 1
                min_rep, max_rep = 1, 1
                if i + 1 < L:
                    nt, nval = tokens[i + 1]
                    if nt == "QMARK":
                        min_rep, max_rep = 0, 1
                        i += 1
                    elif nt == "BRACES":
                        min_rep, max_rep = nval
                        i += 1
                nodes.append(FileNode(paths, min_rep, max_rep))
            else:
                raise ExprError(f"unexpected token during parsing: {t!r}.", i)
            i += 1
        return nodes

    def _combine_recursive(
        self, nodes: list[Node], idx: int = 0
    ) -> Generator[str, None, None]:
        if idx >= len(nodes):
            yield ""
            return
        first = nodes[idx]
        for part in first.expand():
            for suffix in self._combine_recursive(nodes, idx + 1):
                yield part + suffix

    def generate(
        self, nodes: list[Node | FileNode], start_from: str | None = None
    ) -> Generator[str, None, None]:
        gen = self._combine_recursive(nodes, 0)
        if start_from is None:
            yield from gen
            return

        found = False
        for s in gen:
            if not found:
                if s == start_from:
                    found = True
                    yield s
                else:
                    continue
            else:
                yield s

    def _stats_from_nodes(
        self, nodes: list[Node | FileNode], sep_len: int = 1
    ) -> tuple[int, int]:
        total_count = 1
        total_bytes = 0

        for node in nodes:
            if isinstance(node, FileNode):
                k, sum_len_choices = node.stats_info()
                lens = None
            else:
                base = node.base
                if isinstance(base, list):
                    choices = [str(x) for x in base]
                else:
                    choices = [str(base)]
                k = len(choices)
                lens = [len(s.encode("utf-8")) for s in choices]
                sum_len_choices = sum(lens)

            min_r = node.min_rep
            max_r = node.max_rep

            node_count = 0
            node_bytes = 0

            for r in range(min_r, max_r + 1):
                if r == 0:
                    count_r = 1
                    bytes_r = 0
                else:
                    count_r = pow(k, r)
                    bytes_r = r * pow(k, r - 1) * sum_len_choices

                node_count += count_r
                node_bytes += bytes_r

            new_count = total_count * node_count
            new_bytes = total_bytes * node_count + node_bytes * total_count

            total_count, total_bytes = new_count, new_bytes

        return int(total_bytes + (sep_len * total_count)), int(total_count)

    def stats(self, nodes: list[Node], sep_len: int = 1) -> tuple[int, int]:
        return self._stats_from_nodes(nodes, sep_len=sep_len)

    def _node_count(self, node: Node | FileNode) -> int:
        if isinstance(node, FileNode):
            choices = node._collect_lines()
            k = len(choices)
        else:
            base = node.base
            if isinstance(base, list):
                choices = [str(x) for x in base]
            else:
                choices = [str(base)]
            k = len(choices)

        total = 0
        for r in range(node.min_rep, node.max_rep + 1):
            total += 1 if r == 0 else k**r
        return total
