#!/usr/bin/env python
# -*- coding: utf-8 -*-
import threading
import sys
import re

from time import perf_counter
from logging import ERROR
from dataclasses import dataclass

from fuse import __version__
from fuse.logger import log
from fuse.console import get_progress
from fuse.args import create_parser

from fuse.utils.files import r_open
from fuse.utils.formatters import format_size, format_time, parse_size
from fuse.utils.generator import Gen, Node, ExprError


@dataclass
class Progress:
    value: float = 0
    ready: bool = True


def generate(
    generator: Gen,
    nodes: list[Node],
    stats: tuple[int, int],
    buffering: int = 0,
    filename: str | None = None,
    quiet_mode: bool = False,
    sep: str = "\n",
    wrange: tuple[str | None, str | None] = (None, None),
) -> int:
    """Function to generate words"""
    progress = Progress()
    total_bytes, total_words = stats

    event = threading.Event()
    thread = threading.Thread(
        target=get_progress, args=(event, progress), kwargs={"total": total_bytes}
    )
    show_progress_bar = (filename is not None) and (not quiet_mode)

    # uses sys.stdout if filename = None
    with r_open(filename, "a", encoding="utf-8", buffering=buffering) as fp:
        if not fp:
            return 1
        start, end = wrange
        ready = start is None
        progress.ready = ready
        # ignore progress bar to stdout
        if show_progress_bar:
            thread.start()
        start_time = perf_counter()

        def _stop_event() -> None:
            if show_progress_bar:
                event.set()
                thread.join()

        try:
            for _ in generator.generate(nodes, start_from=start):
                if not progress.ready:
                    progress.ready = True
                progress.value += fp.write(_ + sep)
                if end:
                    if end == _:
                        _stop_event()
                        log.warning(f"Wordlist was stopped at '{end}' (--to).")
                        return 0
        except KeyboardInterrupt:
            _stop_event()
            log.warning("Generation stopped with keyboard interrupt!")
            return 0
        if not progress.ready:
            _stop_event()
            log.warning(f"Word '{start}' not found in wordlist generation.")
            return 0
        elapsed = perf_counter() - start_time
        _stop_event()

    if show_progress_bar:
        thread.join()

    log.info(
        f"✨ Complete word generation in {format_time(elapsed)} ({int(total_words/elapsed):,} W/s)."
    )

    return 0


def f_expression(expression: str, files: list) -> tuple[str, list]:
    """Formats string to allow inline expressions and files"""
    n = 0
    i_count = 0

    def i_replace(m: re.Match) -> str:
        nonlocal i_count

        i_count += 1
        if i_count == n:
            return i_str
        return m.group(0)

    i = 0
    files_copy = files.copy()

    def escape_expr(m: re.Match) -> str:
        b = m.group(1)
        if len(b) % 2 == 0:
            return b + r"\@"
        else:
            return m.group(0)

    expression = re.sub(r"(\\*)@", escape_expr, expression)

    for file in files:
        if file.startswith("//"):
            i_str = file.replace("//", "", count=1)
            n += 1
            expression = re.sub(r"(?<!\\)\^", i_replace, expression, count=1)
            files_copy.pop(i)
            i -= 1
        else:
            expression = re.sub(r"(?<!\\)\^", "@", expression, count=1)
        i += 1

    return expression, files_copy


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    if (args.expression is None) and (args.expr_file is None):
        parser.print_help(sys.stderr)
        return 1

    if args.quiet:
        log.setLevel(ERROR)

    if args.buffer.upper() != "AUTO":
        try:
            buffer = parse_size(args.buffer)
            if buffer <= 0:
                raise ValueError("the value cannot be <= 0")
        except ValueError as e:
            log.error(f"invalid buffer size: {e}")
            return 1
    else:
        buffer = -1

    expression = args.expression
    generator = Gen()

    if args.expr_file is not None:
        if args.start or args.end:
            log.error("--from/--to are not supported with expression files.")
            return 1
        with r_open(args.expr_file, "r", encoding="utf-8") as fp:
            if fp is None:
                return 1
            lines = [_.strip() for _ in fp]
            aliases: list[tuple] = []
            files: list[str] = []
            log.info(f'Opening file "{args.expr_file}" with {len(lines)} lines.')
            for i, expression in enumerate(lines):
                if not expression:
                    continue
                for alias in aliases:
                    expression = re.sub(r"(?<!\\)\$" + alias[0], alias[1], expression)
                fields = expression.split(" ")
                if fields[0] == "#":  # ignore comments
                    continue
                if fields[0] == r"%alias":
                    if len(fields) < 3:
                        log.error(
                            r"Invalid File: '%alias' keyword requires 2 arguments."
                        )
                        return 1
                    alias = fields[1].strip()
                    alias_value = " ".join(fields[2:])
                    aliases.append((alias, alias_value))
                    continue
                elif fields[0] == r"%file":
                    if len(fields) < 2:
                        log.error(
                            r"Invalid File: '%file' keyword requires 1 arguments."
                        )
                        return 1
                    files.append(" ".join(fields[1:]).strip())
                    continue
                try:
                    tokens = generator.tokenize(expression)
                    nodes = generator.parse(tokens, files=(files or None))
                    s_bytes, s_words = generator.stats(
                        nodes, sep_len=len(args.separator)
                    )
                    files = []
                except ExprError as e:
                    log.error(e)
                    return 1
                log.info(
                    f"Generating {s_words:,} words ({format_size(s_bytes)}) for L{i+1}..."
                )
                c = generate(
                    generator,
                    nodes,
                    (s_bytes, s_words),
                    filename=args.output,
                    buffering=buffer,
                    quiet_mode=args.quiet,
                    sep=args.separator,
                )
                if c != 0:
                    return c
        return 0

    if args.end is not None:
        log.warning("Using --to: wordlist generation should stop before completion.")

    expression, files = f_expression(expression, args.files)

    try:
        tokens = generator.tokenize(expression)
        nodes = generator.parse(tokens, files=(files or None))
        s_bytes, s_words = generator.stats(nodes, sep_len=len(args.separator))
    except ExprError as e:
        log.error(e)
        return 1

    log.info(f"Fuse Version @ {__version__}")
    log.info(f"Fuse will generate {s_words:,} words ({format_size(s_bytes)}).")

    if not args.quiet:
        while True:
            try:
                r = input("[Y/n] Continue? ").upper()
            except KeyboardInterrupt:
                return 0

            if not r or r == "Y":
                break
            elif r == "N":
                return 0
            else:
                log.info('Please answer "Y" or "N"...')

    sys.stdout.write("\n")
    sys.stdout.flush()

    return generate(
        generator,
        nodes,
        (s_bytes, s_words),
        filename=args.output,
        buffering=buffer,
        quiet_mode=args.quiet,
        sep=args.separator,
        wrange=(args.start, args.end),
    )
