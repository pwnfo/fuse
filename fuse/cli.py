#!/usr/bin/env python
# -*- coding: utf-8 -*-
import threading
import sys
import re

from time import perf_counter
from logging import ERROR
from dataclasses import dataclass

from .console import log, get_progress, console_input
from .args import create_parser

from .utils.misc import r_open, format_size, format_time
from .utils.generator import Gen, Node, ExprError


@dataclass
class Progress:
    value: float = 0


def generate(
    generator: Gen,
    nodes: list[Node],
    total_bytes: int,
    buffering: int = 0,
    filename: str | None = None,
    quiet_mode: bool = False,
    sep: str = "\n",
) -> int:
    """Function to generate words"""
    progress = Progress()

    event = threading.Event()
    thread = threading.Thread(
        target=get_progress, args=(event, progress), kwargs={"total": total_bytes}
    )
    show_progress_bar = (filename is not None) and (not quiet_mode)

    # uses sys.stdout if filename = None
    with r_open(filename, "a", encoding="utf-8", buffering=buffering) as fp:
        if fp:
            # ignore progress bar to stdout
            if show_progress_bar:
                thread.start()
            start_time = perf_counter()
            try:
                for _ in generator.generate(nodes):
                    progress.value += fp.write(_ + sep)
            except KeyboardInterrupt:
                if show_progress_bar:
                    event.set()
                    thread.join()
                log.info("Goodbye!")
                return 0
            elapsed = perf_counter() - start_time
        else:
            return 1

    if show_progress_bar:
        thread.join()

    log.info(
        f"Complete word generation in {format_time(elapsed)} ({int(total_bytes/elapsed)} W/s)."
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

    expression = args.expression
    generator = Gen()

    if args.expr_file is not None:
        with r_open(args.expr_file, "r", encoding="utf-8") as fp:
            if fp is None:
                return 1
            lines = [_.strip() for _ in fp]
            aliases: list[tuple] = []
            files: list[str] = []
            log.info(f'Opening file "{args.expr_file}" with {len(lines)} lines.')
            for i, expression in enumerate(lines):
                expression = expression.split("#")[0].strip()  # ignore comments
                if not expression:
                    continue
                for alias in aliases:
                    expression = re.sub(r"(?<!\\)\$" + alias[0], alias[1], expression)
                if expression.startswith(r"%alias "):
                    fields = expression.split(" ")
                    if len(fields) < 3:
                        log.error(
                            r"Invalid File: '%alias' keyword requires 2 arguments."
                        )
                        return 1
                    alias = fields[1].strip()
                    alias_value = " ".join(fields[2:])
                    aliases.append((alias, alias_value))
                    continue
                elif expression.startswith(r"%file "):
                    fields = expression.split(" ")
                    if len(fields) < 2:
                        log.error(
                            r"Invalid File: '%file' keyword requires 1 arguments."
                        )
                        return 1
                    files.append(fields[1].strip())
                    continue
                try:
                    tokens = generator.tokenize(expression)
                    nodes = generator.parse(tokens, files=(files or None))
                    s_bytes, s_words = generator.stats(nodes)
                    files = []
                except ExprError as e:
                    log.error(f"Expression Error: {e}")
                    return 1
                log.info(
                    f"Generating {s_words} words ({format_size(s_bytes)}) for L{i+1}..."
                )
                c = generate(
                    generator,
                    nodes,
                    total_bytes=s_bytes,
                    filename=args.output,
                    buffering=args.buffer,
                    quiet_mode=args.quiet,
                    sep=args.separator,
                )
                if c != 0:
                    return c
        return 0

    expression, files = f_expression(expression, args.files)

    try:
        tokens = generator.tokenize(expression)
        nodes = generator.parse(tokens, files=(files or None))
        s_bytes, s_words = generator.stats(nodes)
    except ExprError as e:
        log.error(f"expression error: {e}")
        return 1

    log.info(f"Fuse will generate {s_words} words ({format_size(s_bytes)}).")

    if not args.quiet:
        try:
            r = console_input(
                "[Y/n] Continue? ", fprompt="[cyan]\\[Y/n][/cyan] Continue? "
            )
        except KeyboardInterrupt:
            return 0

        if r.lower() == "n":
            return 0

    return generate(
        generator,
        nodes,
        s_bytes,
        filename=args.output,
        buffering=args.buffer,
        quiet_mode=args.quiet,
        sep=args.separator,
    )
