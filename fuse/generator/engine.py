import re
from itertools import product
from typing import Generator, Any

from fuse.utils.classes import pattern_repl
from fuse.generator.exceptions import ExprError
from fuse.generator.nodes import Node, FileNode, BindDefNode, BindRefNode

class FuseGenerator:
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

    def _find_binding_close(self, s: str, start: int) -> int:
        """Find the closing '>' of a <@...> binding, correctly skipping nested brackets."""
        i = start
        n = len(s)
        depth_square = 0
        depth_paren = 0
        depth_brace = 0
        while i < n:
            ch = s[i]
            if ch == "\\":
                i += 2
                continue
            if ch == "[":
                depth_square += 1
            elif ch == "]":
                if depth_square > 0:
                    depth_square -= 1
            elif ch == "(":
                depth_paren += 1
            elif ch == ")":
                if depth_paren > 0:
                    depth_paren -= 1
            elif ch == "{":
                depth_brace += 1
            elif ch == "}":
                if depth_brace > 0:
                    depth_brace -= 1
            elif (
                ch == ">"
                and depth_square == 0
                and depth_paren == 0
                and depth_brace == 0
            ):
                return i
            i += 1
        return -1

    def _parse_range(self, pattern: str, start_idx: int) -> tuple[list[str], int]:
        end_pos = self._find_closing(pattern, start_idx, "]")
        if end_pos == -1:   
            raise ExprError("unclosed range (missing ']')", error_pos=(pattern, start_idx))
        inner = pattern[start_idx:end_pos]
        m = self.RANGE_RE.match(inner)
        if not m:
            raise ExprError("invalid range syntax (expected '#[START-END[:STEP]]')", error_pos=(pattern, start_idx))
        r_start = int(m.group(1))
        r_end = int(m.group(2))
        step_str = m.group(3)
        step = int(step_str) if step_str else (1 if r_start <= r_end else -1)
        if step == 0:
            raise ExprError("range step cannot be zero", error_pos=(pattern, start_idx))
        if r_start < 0 or r_end < 0:
            raise ExprError("range bounds must be non-negative", error_pos=(pattern, start_idx))
        if (step > 0 and r_start > r_end) or (step < 0 and r_start < r_end):
            raise ExprError("invalid range sequence", error_pos=(pattern, start_idx))
        if step > 0:
            rng = range(r_start, r_end + 1, step)
        else:
            rng = range(r_start, r_end - 1, step)
        choices = [str(x) for x in rng]
        if not choices:
            raise ExprError("range produced no values", error_pos=(pattern, start_idx))
        return choices, end_pos + 1

    def _parse_class(
        self, pattern: str, start_idx: int, literal_mode: bool
    ) -> tuple[list[str], int]:
        closer = ")" if literal_mode else "]"
        end_pos = self._find_closing(pattern, start_idx, closer)
        if end_pos == -1:
            raise ExprError(f"unclosed character class (missing {closer!r})", error_pos=(pattern, start_idx))
        inner = pattern[start_idx:end_pos]
        if not inner:
            raise ExprError("empty character class is not allowed", error_pos=(pattern, start_idx))
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
            raise ExprError("invalid character class contents", error_pos=(pattern, start_idx))
        return choices, end_pos + 1

    def _tokenize_raw(self, pattern: str) -> list[tuple[str, Any]]:
        """Core tokenization loop (pattern_repl must already have been applied)."""
        i = 0
        n = len(pattern)
        tokens: list[tuple[str, Any]] = []
        pr = pattern
        BR = self.BRACES_RE
        while i < n:
            c = pr[i]
            if c == "\\":
                if i + 1 >= n:
                    raise ExprError("invalid escape sequence (trailing backslash)", error_pos=(pattern, i+1))
                tokens.append(("LIT", pr[i + 1]))
                i += 2
                continue
            if c == "<":
                if i + 1 < n and pr[i + 1] == "@":
                    end = self._find_binding_close(pr, i + 2)
                    if end == -1:
                        raise ExprError("unclosed binding (missing '>')", error_pos=(pattern, i+1))
                    inner = pr[i + 2 : end]
                    eq_pos = inner.find("=")
                    if eq_pos == -1:
                        name = inner.strip()
                        if not name.isidentifier():
                            raise ExprError(f"invalid binding name {name!r}", error_pos=(pattern, i+1))
                        tokens.append(("BIND_REF", name))
                    else:
                        name = inner[:eq_pos].strip()
                        expr = inner[eq_pos + 1 :]
                        if not name.isidentifier():
                            raise ExprError(f"invalid binding name {name!r}", error_pos=(pattern, i+1))
                        inner_tokens = self._tokenize_raw(expr)
                        tokens.append(("BIND_DEF", (name, inner_tokens)))
                    i = end + 1
                    continue
                tokens.append(("LIT", "<"))
                i += 1
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
            if c == "^":
                tokens.append(("FILE", None))
                i += 1
                continue
            if c == "{":
                m = BR.match(pr[i:])
                if m:
                    a = int(m.group(1))
                    b = int(m.group(2)) if m.group(2) is not None else a
                    if a > b:
                        raise ExprError("invalid repetition range (min > max)", error_pos=(pattern, i+1))
                    tokens.append(("BRACES", (a, b)))
                    i += m.end()
                    continue
                else:
                    raise ExprError("invalid repetition syntax", error_pos=(pattern, i+1))
            tokens.append(("LIT", c))
            i += 1
        return tokens

    def tokenize(self, pattern: str) -> list[tuple[str, Any]]:
        return self._tokenize_raw(pattern_repl(pattern))

    def _parse_tokens(
        self,
        tokens: list[tuple[str, Any]],
        file_groups: list[list[str]],
        file_idx: int,
    ) -> tuple[list, int]:
        """Parse a flat token list into nodes, returning (nodes, updated_file_idx)."""
        nodes: list = []
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
                    raise ExprError("insufficient file assignments")
                nodes.append(FileNode(file_groups[file_idx], min_rep, max_rep))
                file_idx += 1
            elif kind == "BIND_DEF":
                name, inner_tokens = val
                inner_nodes, file_idx = self._parse_tokens(
                    inner_tokens, file_groups, file_idx
                )
                nodes.append(BindDefNode(name, inner_nodes, min_rep, max_rep))
            elif kind == "BIND_REF":
                nodes.append(BindRefNode(val, min_rep, max_rep))
            else:
                raise ExprError(f"unexpected token {kind!r}")
            i += 1
        return nodes, file_idx

    def _count_file_tokens(self, tokens: list[tuple[str, Any]]) -> int:
        """Recursively count FILE tokens, including those inside BIND_DEF inner_tokens."""
        count = 0
        for kind, val in tokens:
            if kind == "FILE":
                count += 1
            elif kind == "BIND_DEF":
                _, inner_tokens = val
                count += self._count_file_tokens(inner_tokens)
        return count

    def parse(
        self, tokens: list[tuple[str, Any]], files: list[str] | None = None
    ) -> list:
        count_ft = self._count_file_tokens(tokens)
        if count_ft:
            if not files:
                raise ExprError("pattern requires files but none were provided")
            if count_ft == 1:
                file_groups: list[list[str]] = [files]
            else:
                if len(files) < count_ft:
                    raise ExprError(
                        f"pattern requires {count_ft} files but {len(files)} were provided"
                    )
                file_groups = [[f] for f in files[:count_ft]]
        else:
            file_groups = []
        nodes, _ = self._parse_tokens(tokens, file_groups, 0)
        return nodes

    def _combine_resume(
        self,
        nodes: list,
        idx: int,
        start_from: str | None,
        bindings: dict[str, str] | None = None,
    ) -> Generator[str, None, None]:
        """recursive word combination generator with resume logic and binding support."""
        if bindings is None:
            bindings = {}
        ln = len(nodes)
        if idx >= ln:
            if not start_from:
                yield ""
            return
        cur = nodes[idx]

        if isinstance(cur, BindDefNode):
            inner_vals: list[str]
            if cur.min_rep == 1 and cur.max_rep == 1:
                inner_vals_gen = self._combine_resume(cur.inner_nodes, 0, None, {})
            else:
                base_vals = list(self._combine_resume(cur.inner_nodes, 0, None, {}))

                def _rep_gen(
                    base: list[str], mn: int, mx: int
                ) -> Generator[str, None, None]:
                    for r in range(mn, mx + 1):
                        if r == 0:
                            yield ""
                        else:
                            for combo in product(base, repeat=r):
                                yield "".join(combo)

                inner_vals_gen = _rep_gen(base_vals, cur.min_rep, cur.max_rep)
            for val in inner_vals_gen:
                new_bindings = {**bindings, cur.name: val}
                for suffix in self._combine_resume(nodes, idx + 1, None, new_bindings):
                    yield val + suffix
            return

        if isinstance(cur, BindRefNode):
            if cur.name not in bindings:
                raise ExprError(f"undefined variable {cur.name!r}")
            val_base = bindings[cur.name]
            for r in range(cur.min_rep, cur.max_rep + 1):
                val = val_base * r if r > 0 else ""
                if start_from is None:
                    next_target = None
                elif start_from.startswith(val):
                    next_target = start_from[len(val) :]
                elif val.startswith(start_from):
                    next_target = None
                else:
                    continue
                for suffix in self._combine_resume(
                    nodes, idx + 1, next_target, bindings
                ):
                    yield val + suffix
            return

        if start_from is None:
            for part in cur.expand():
                for suffix in self._combine_resume(nodes, idx + 1, None, bindings):
                    yield part + suffix
            return

        for part, remainder, is_full_mode in cur.expand_resume(start_from):
            next_target = None if is_full_mode else remainder
            for suffix in self._combine_resume(nodes, idx + 1, next_target, bindings):
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
                if item >= start_from:
                    found = True
                else:
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

    def _calculate_skipped_count(self, nodes: list, target: str) -> int:
        """calculates how many words exist strictly before 'target'."""
        skipped_count = 0
        current_target = target
        bindings: dict[str, str] = {}

        for i, node in enumerate(nodes):
            if isinstance(node, BindDefNode):
                inner_idx = 0
                found_val: str | None = None
                for val in self._combine_resume(node.inner_nodes, 0, None, {}):
                    if current_target.startswith(val):
                        found_val = val
                        break
                    inner_idx += 1
                if found_val is None:
                    break
                suffix_capacity = self._get_suffix_capacity(nodes, i + 1)
                skipped_count += inner_idx * suffix_capacity
                bindings[node.name] = found_val
                current_target = current_target[len(found_val) :]
                if not current_target and i < len(nodes) - 1:
                    break
            elif isinstance(node, BindRefNode):
                if node.name not in bindings:
                    raise ExprError(f"undefined variable {node.name!r}")
                val = bindings[node.name]
                if node.min_rep != node.max_rep:
                    ref_card = node.cardinality
                    suffix_capacity = self._get_suffix_capacity(nodes, i + 1)
                    for r in range(node.min_rep, node.max_rep + 1):
                        out = val * r if r > 0 else ""
                        if current_target.startswith(out):
                            node_skipped = r - node.min_rep
                            skipped_count += node_skipped * suffix_capacity
                            current_target = current_target[len(out) :]
                            break
                else:
                    out = val * node.min_rep if node.min_rep > 0 else ""
                    if not current_target.startswith(out):
                        break
                    current_target = current_target[len(out) :]
                if not current_target and i < len(nodes) - 1:
                    break
            else:
                node_skipped, remainder = node.get_skipped_count(current_target)
                suffix_capacity = self._get_suffix_capacity(nodes, i + 1)
                skipped_count += node_skipped * suffix_capacity
                if remainder is None:
                    break
                current_target = remainder
                if not current_target and i < len(nodes) - 1:
                    break

        return skipped_count

    def get_word_at_index(self, nodes: list, index: int) -> str:
        """retrieves the word at a specific index."""
        result: list[str] = []
        bindings: dict[str, str] = {}
        for i, node in enumerate(nodes):
            suffix_cap = self._get_suffix_capacity(nodes, i + 1)
            if isinstance(node, BindDefNode):
                inner_vals = list(self._combine_resume(node.inner_nodes, 0, None, {}))
                node_idx = index // suffix_cap
                index %= suffix_cap
                val = inner_vals[node_idx]
                bindings[node.name] = val
                result.append(val)
            elif isinstance(node, BindRefNode):
                if node.name not in bindings:
                    raise ExprError(f"undefined variable {node.name!r}")
                val_base = bindings[node.name]
                if node.min_rep != node.max_rep:
                    node_idx = index // suffix_cap
                    index %= suffix_cap
                    r = node.min_rep + node_idx
                    result.append(val_base * r if r > 0 else "")
                else:
                    result.append(val_base * node.min_rep if node.min_rep > 0 else "")
            else:
                node_idx = index // suffix_cap
                index %= suffix_cap
                result.append(node.get_item_at(node_idx))
        return "".join(result)

    def stats(
        self,
        nodes: list,
        delimiter_len: int = 1,
        start_from: str | None = None,
        end: str | None = None,
    ) -> tuple[int, int]:
        """calculates wordlist stats, adjusted for start_from and end range."""
        total_count = 1
        total_bytes = 0
        binding_stats: dict[str, tuple[int, int]] = {}

        for node in nodes:
            if isinstance(node, BindDefNode):
                inner_bytes, inner_count = self.stats(node.inner_nodes, delimiter_len=0)
                node_count = inner_count
                node_bytes = inner_bytes
                avg_len = inner_bytes // inner_count if inner_count > 0 else 0
                binding_stats[node.name] = (inner_count, avg_len)
            elif isinstance(node, BindRefNode):
                if node.name not in binding_stats:
                    raise ExprError(f"undefined variable {node.name!r}")
                _, avg_len = binding_stats[node.name]
                node_count = 1
                node_bytes = avg_len
            elif isinstance(node, FileNode):
                k, sum_len = node.stats_info()
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
                            node_count += 1
                        else:
                            node_count += k**r
                            node_bytes += r * (k ** (r - 1)) * sum_len
            else:
                choices = node.base
                k = len(choices)
                cached = node._sum_len
                if cached is None:
                    s = sum(len(str(s_item).encode("utf-8")) for s_item in choices)
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
                            node_count += 1
                        else:
                            node_count += k**r
                            node_bytes += r * (k ** (r - 1)) * sum_len
            total_count, total_bytes = total_count * node_count, (
                total_bytes * node_count
            ) + (node_bytes * total_count)

        full_total_bytes = int(total_bytes + (delimiter_len * total_count))
        full_total_count = int(total_count)

        if not start_from and not end:
            return full_total_bytes, full_total_count

        start_deduction = 0
        if start_from:
            start_idx = self._calculate_skipped_count(nodes, start_from)
            start_deduction = start_idx

        end_cap = full_total_count
        if end:
            end_idx = self._calculate_skipped_count(nodes, end)
            end_cap = end_idx + 1

        actual_count = end_cap - start_deduction
        if actual_count < 0:
            actual_count = 0

        if full_total_count > 0:
            ratio = actual_count / full_total_count
            actual_bytes = int(full_total_bytes * ratio)
        else:
            actual_bytes = 0

        return actual_bytes, actual_count
