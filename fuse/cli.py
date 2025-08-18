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
from .utils.interpreter import Interpreter, Node, ExprError


@dataclass
class Progress:
    value: float = 0


def generate(
    interp: Interpreter,
    nodes: list[Node],
    total_bytes: int,
    buffering: int = 0,
    filename: str | None = None,
    quiet_mode: bool = False,
) -> int:
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
                for _ in interp.generate(nodes):
                    progress.value += fp.write(_ + "\n")
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
    i = 0
    expression = re.sub(r"(?<!\\)@", "\\@", expression, count=1)
    files_copy = files.copy()
    for file in files:
        if file.startswith("//"):
            _ = file.replace("//", "", count=1).replace("^", "\\^")
            expression = re.sub(r"(?<!\\)\^", _, expression, count=1)
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
    interp = Interpreter()

    if args.expr_file is not None:
        with r_open(args.expr_file, "r", encoding="utf-8") as fp:
            if fp is None:
                return 1
            lines = [_.strip() for _ in fp]
            log.info(f'Opening file "{args.expr_file}" with {len(lines)} expressions.')
            for i, expression in enumerate(lines):
                if (expression == "") or (expression.startswith("# ")):
                    continue
                try:
                    tokens = interp.tokenize(expression)
                    nodes = interp.parse(tokens)
                    s_bytes, s_words = interp.stats(nodes)
                except ExprError as e:
                    log.error(f"expression error: {e}")
                    return 1
                log.info(
                    f"Generating {s_words} words ({format_size(s_bytes,d=0)}) for L{i+1}..."
                )
                c = generate(
                    interp,
                    nodes,
                    total_bytes=s_bytes,
                    filename=args.output,
                    buffering=args.buffer,
                    quiet_mode=args.quiet,
                )
                if c != 0:
                    return c
        return 0

    expression, files = f_expression(expression, args.files)

    try:
        tokens = interp.tokenize(expression)
        nodes = interp.parse(tokens, files=files or None)
        s_bytes, s_words = interp.stats(nodes)
    except ExprError as e:
        log.error(f"expression error: {e}")
        return 1

    log.info(f"Fuse will generate {s_words} words ({format_size(s_bytes,d=0)}).")

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
        interp,
        nodes,
        s_bytes,
        filename=args.output,
        buffering=args.buffer,
        quiet_mode=args.quiet,
    )
