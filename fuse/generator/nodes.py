from itertools import product
from typing import Generator
from fuse.utils.files import fuse_open
from fuse.generator.exceptions import ExprError


class BindDefNode:
    """Represents <@name=expr>: generates values from inner_nodes and stores each under name."""

    __slots__ = ("name", "inner_nodes", "min_rep", "max_rep", "_cached_cardinality")

    def __init__(
        self, name: str, inner_nodes: list, min_rep: int = 1, max_rep: int = 1
    ) -> None:
        self.name = name
        self.inner_nodes = inner_nodes
        self.min_rep = min_rep
        self.max_rep = max_rep
        self._cached_cardinality: int | None = None

    def __repr__(self) -> str:
        return f"<BindDefNode name={self.name!r} {{{self.min_rep},{self.max_rep}}} inner={self.inner_nodes!r}>"

    @property
    def cardinality(self) -> int:
        if self._cached_cardinality is not None:
            return self._cached_cardinality
        inner_card = 1
        for n in self.inner_nodes:
            inner_card *= n.cardinality
        if self.min_rep == 1 and self.max_rep == 1:
            total = inner_card
        else:
            total = 0
            for r in range(self.min_rep, self.max_rep + 1):
                total += 1 if r == 0 else inner_card**r
        self._cached_cardinality = total
        return total


class BindRefNode:
    """Represents <@name>: outputs the value previously stored under name."""

    __slots__ = ("name", "min_rep", "max_rep")

    def __init__(self, name: str, min_rep: int = 1, max_rep: int = 1) -> None:
        self.name = name
        self.min_rep = min_rep
        self.max_rep = max_rep

    def __repr__(self) -> str:
        return f"<BindRefNode name={self.name!r} {{{self.min_rep},{self.max_rep}}}>"

    @property
    def cardinality(self) -> int:
        """Number of distinct outputs (one per repetition count in [min_rep, max_rep])."""
        return self.max_rep - self.min_rep + 1


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
        join = "".join
        for k in range(min_r, max_r + 1):
            if k == 0:
                yield ""
            else:
                for tup in product(base, repeat=k):
                    yield join(tup)

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

    def get_skipped_stats(
        self, target: str, current_prefix_len: int = 0
    ) -> tuple[int, int, str | None]:
        """calculates (count, bytes, remainder) strictly before 'target'."""
        skipped_count = 0
        skipped_bytes = 0
        base_len = len(self.base)

        if self._sum_len is None:
            self._sum_len = sum(len(str(x).encode("utf-8")) for x in self.base)

        sum_len = self._sum_len

        for k in range(self.min_rep, self.max_rep + 1):
            if k == 0:
                if target:
                    skipped_count += 1
                    continue
                else:
                    return skipped_count, skipped_bytes, ""

            res_count, res_bytes, res_remainder = self._calc_skip_recursive_stats(
                self.base, k, target, current_prefix_len
            )

            if res_remainder is not None:
                return (
                    skipped_count + res_count,
                    skipped_bytes + res_bytes,
                    res_remainder,
                )

            skipped_count += base_len**k
            level_bytes = (k * (base_len ** (k - 1)) * sum_len) + (
                (base_len**k) * current_prefix_len
            )
            skipped_bytes += level_bytes

        return skipped_count, skipped_bytes, None

    def _calc_skip_recursive_stats(
        self, pool: list[str], depth: int, target: str, prefix_len: int
    ) -> tuple[int, int, str | None]:
        """recursive helper for skipped stats (count, bytes, remainder)."""
        if depth == 0:
            return 0, 0, target

        skipped_c = 0
        skipped_b = 0
        pool_len = len(pool)

        if self._sum_len is None:
            self._sum_len = sum(len(str(x).encode("utf-8")) for x in pool)
        sum_len = self._sum_len

        for item in pool:
            item_b = len(item.encode("utf-8"))
            if target.startswith(item):
                rem_target = target[len(item) :]
                rec_c, rec_b, rec_rem = self._calc_skip_recursive_stats(
                    pool, depth - 1, rem_target, prefix_len + item_b
                )

                if rec_rem is not None:
                    return skipped_c + rec_c, skipped_b + rec_b, rec_rem
                else:
                    skipped_c += pool_len ** (depth - 1)
                    branch_b = (
                        (depth - 1) * (pool_len ** (max(0, depth - 2))) * sum_len
                    ) + ((pool_len ** (depth - 1)) * (prefix_len + item_b))
                    skipped_b += branch_b

            elif item.startswith(target):
                return skipped_c, skipped_b, ""

            else:
                skipped_c += pool_len ** (depth - 1)
                branch_b = (
                    (depth - 1) * (pool_len ** (max(0, depth - 2))) * sum_len
                ) + ((pool_len ** (depth - 1)) * (prefix_len + item_b))
                skipped_b += branch_b

        return skipped_c, skipped_b, None

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
                with fuse_open(path, "r", encoding="utf-8", errors="ignore") as fp:
                    if not fp:
                        raise IOError
                    out.extend(ln.rstrip("\n\r") for ln in fp)
            except (IOError, OSError):
                raise ExprError(f"failed to open or read file {path!r}")
        if not out:
            raise ExprError(f"no lines produced from files {self.base!r}")
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

    def get_skipped_stats(
        self, target: str, current_prefix_len: int = 0
    ) -> tuple[int, int, str | None]:
        """calculates skipped count and bytes using file lines as base."""
        skipped_count = 0
        skipped_bytes = 0
        choices = self.lines
        base_len = len(choices)

        count_info, sum_len = self.stats_info()

        for k in range(self.min_rep, self.max_rep + 1):
            if k == 0:
                if target:
                    skipped_count += 1
                    continue
                else:
                    return skipped_count, skipped_bytes, ""

            res_count, res_bytes, res_remainder = self._calc_skip_recursive_stats(
                choices, k, target, current_prefix_len
            )
            if res_remainder is not None:
                return (
                    skipped_count + res_count,
                    skipped_bytes + res_bytes,
                    res_remainder,
                )

            skipped_count += base_len**k
            level_bytes = (k * (base_len ** (k - 1)) * sum_len) + (
                (base_len**k) * current_prefix_len
            )
            skipped_bytes += level_bytes

        return skipped_count, skipped_bytes, None

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
