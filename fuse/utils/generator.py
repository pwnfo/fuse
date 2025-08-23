import re
import sys

from itertools import product
from typing import Generator, Any, List, Never

from .classes import pattern_repl
from .misc import r_open


class ExprError(Exception): ...


class Node:
    def __init__(self, base: str | list, min_rep: int = 1, max_rep: int = 1) -> None:
        self.base = base
        self.min_rep = min_rep
        self.max_rep = max_rep

    def __repr__(self) -> str:
        return f"<Node base={self.base!r} {{{self.min_rep},{self.max_rep}}}>"

    def expand(self) -> Generator[str, None, None]:
        """Generate all possible Node words"""
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
        """Returns all lines of the file"""
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
        return lines

    def stats_info(self) -> tuple[int, int] | Never:
        """Returns the statistics of this specific file

        Returns:
            tuple[int, int] -> Tuple containing the number of words
                               to be generated (index 0) and total bytes (index 1)
        """
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
        """Generates the words from the file"""
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
        """Transforms the pattern into a list of tokens

        Returns:
            list[tuple[str, Any]] -> list containing the token type
                                     and its value in each index
        """
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
                        "invalid escape: pattern ends with a single backslash '\\'."
                    )
            elif c == "(":
                match = re.search(r"(?<!\\)\]", pattern[i + 1 :])
                if match is None:
                    raise ExprError("unclosed literal class: missing ')'.")
                j = i + 1 + match.start()
                inner = pattern[i + 1 : j]
                if inner == "":
                    raise ExprError("empty literal class '()' is not allowed.")
                tokens.append(("CLASS", [inner]))
                i = j + 1
            elif c == "[":
                match = re.search(r"(?<!\\)\]", pattern[i + 1 :])
                if match is None:
                    raise ExprError("unclosed character class: missing ']'.")
                j = i + 1 + match.start()
                inner = pattern[i + 1 : j]
                if inner == "":
                    raise ExprError("empty character class '[]' is not allowed.")
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
                            "invalid character class contents inside [...]."
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
                        "invalid repetition syntax: expected '{R}' or '{MIN, MAX}'."
                    )
                a = int(m.group(1))
                b = m.group(2)
                if b is None:
                    tokens.append(("BRACES", (a, a)))
                else:
                    b_int = int(b)
                    if a > b_int:
                        raise ExprError(
                            "invalid repetition range: MIN cannot be greater than MAX in '{MIN, MAX}'."
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
        """Transforms a list of tokens into a list of nodes (`Node` and/or `FileNode`)"""
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
                    "pattern contains '@' file placeholder but no files were provided."
                )
            if file_token_count == 1:
                file_assignments = [files]
            else:
                if len(files) < file_token_count:
                    raise ExprError(
                        f"pattern requires {file_token_count} file(s) (one per '@'), but only {len(files)} provided."
                    )
                file_assignments = [[f] for f in files[:file_token_count]]

        current_file_idx = 0

        while i < L:
            t, val = tokens[i]
            if t in ("LIT", "CLASS"):
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
                    raise ExprError("no files assigned for '@' token")
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
                raise ExprError(f"unexpected token during parsing: {t!r}.")
            i += 1
        return nodes

    def _combine_recursive(
        self, nodes: list[Node], idx: int = 0
    ) -> Generator[str, None, None]:
        """Calls the `expand` method of each node"""
        if idx >= len(nodes):
            yield ""
            return
        first = nodes[idx]
        for part in first.expand():
            for suffix in self._combine_recursive(nodes, idx + 1):
                yield part + suffix

    def generate(self, nodes: list[Node | FileNode]) -> Generator[str, None, None]:
        """Call `_combine_recursive` to generate the possible words from the list of `Node`"""
        yield from self._combine_recursive(nodes, 0)

    # def expand_pattern(
    #     self, pattern: str, files: List[str] | None = None
    # ) -> Generator[str, None, None]:
    #     tokens = self.tokenize(pattern)
    #     nodes = self.parse(tokens, files=files)
    #     return self.generate(nodes)

    # def tokens(self, pattern: str) -> list[tuple[str, Any]]:
    #     return self.tokenize(pattern)

    # def nodes(self, pattern: str, files: List[str] | None = None) -> list[Node]:
    #     return self.parse(self.tokenize(pattern), files=files)

    def _stats_from_nodes(self, nodes: list[Node | FileNode]) -> tuple[int, int]:
        """Method that will be called by `stats`"""
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

        return int(total_bytes), int(total_count)

    def stats(self, nodes: list[Node]) -> tuple[int, int]:
        """Generate statistics (number of bytes and words that are generated) for each `Node` or `FileNode`"""
        return self._stats_from_nodes(nodes)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('"PATTERN" is required.')
        sys.exit(1)

    pattern = sys.argv[1]
    files = sys.argv[2:]
    generator = Gen()

    try:
        tokens = generator.tokenize(pattern)
        for _, token in enumerate(tokens):
            print(f"=== TOKEN[{_}] ===", f"\n  TYPE: {token[0]}\n  VALUE: {token[1]}")
        nodes = generator.parse(tokens, files=files if files else None)
        print("------")
        for _, node in enumerate(nodes):
            print(
                f"=== NODE[{_}] ===",
                f"\n  BASE: {node.base}\n  REP: {node.min_rep}-{node.max_rep}",
            )
    except ExprError as e:
        print("ERROR:", e)
        sys.exit(1)

    answer = input("Generate? [y/N] ").strip().lower()
    if answer != "y":
        sys.exit(0)

    try:
        for s in generator.generate(nodes):
            print(s)
    except ExprError as e:
        print("ERROR:", e)
        sys.exit(1)
