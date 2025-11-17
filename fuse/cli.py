#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys
import threading

from dataclasses import dataclass
from datetime import datetime
from getpass import getpass
from logging import ERROR
from time import perf_counter
from typing import List, Tuple, Optional
from fuse import __version__

from fuse.args import create_parser
from fuse.console import get_progress
from fuse.logger import log
from fuse.utils.files import r_open
from fuse.utils.formatters import format_size, format_time, parse_size
from fuse.utils.generator import ExprError, Node, WordlistGenerator


@dataclass
class Progress:
    value: float = 0


def generate(
    generator: WordlistGenerator,
    nodes: List[Node],
    stats: Tuple[int, int],
    buffering: int = 0,
    filename: Optional[str] = None,
    quiet_mode: bool = False,
    sep: str = "\n",
    wrange: Tuple[Optional[str], Optional[str]] = (None, None),
) -> int:
    progress = Progress()
    total_bytes, total_words = stats

    # thread for progress bar
    event = threading.Event()
    thread = threading.Thread(
        target=get_progress, args=(event, progress), kwargs={"total": total_bytes}
    )
    show_progress_bar = (filename is not None) and (not quiet_mode)

    # output file or stdout
    with r_open(filename, "a", encoding="utf-8", buffering=buffering) as fp:
        if not fp:
            return 1

        start_token, end_token = wrange

        if show_progress_bar:
            thread.start()

        start_time = perf_counter()

        # stops progress thread
        def stop_progress() -> None:
            if show_progress_bar and not event.is_set():
                event.set()
                thread.join()

        log.info(
            datetime.now().strftime(
                "Starting wordlist generation at %H:%M:%S on %a %b %d %Y."
            )
        )

        try:
            for token in generator.generate(nodes, start_from=start_token):
                progress.value += fp.write(token + sep)

                # stop when reaching --to
                if end_token == token:
                    stop_progress()
                    break
        except KeyboardInterrupt:
            stop_progress()
            log.warning("Generation stopped with keyboard interrupt!")

            return 1
        except Exception:
            stop_progress()
            raise

        elapsed = perf_counter() - start_time
        stop_progress()

    if show_progress_bar and thread.is_alive():
        thread.join()

    speed = int(total_words / elapsed) if elapsed > 0 else 0
    log.info(f"Complete word generation in {format_time(elapsed)} ({speed:,} W/s).")

    return 0


def f_expression(expression: str, files: List[str]) -> Tuple[str, List[str]]:
    n_files = 0
    files_out: List[str] = []

    # escapes @
    def escape_expr(m: re.Match) -> str:
        b = m.group(1)
        return b + r"\@" if len(b) % 2 == 0 else m.group(0)

    expression = re.sub(r"(\\*)@", escape_expr, expression)

    for file_path in files:
        if file_path.startswith("//"):
            inline = file_path.replace("//", "", 1)
            expression = re.sub(r"(?<!\\)\^", lambda m: inline, expression, count=1)
            n_files += 1
        else:
            expression = re.sub(r"(?<!\\)\^", "@", expression, count=1)
            files_out.append(file_path)

    return expression, files_out


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    if args.expression is None and args.expr_file is None:
        parser.print_help(sys.stderr)
        return 1

    if args.quiet:
        log.setLevel(ERROR)

    buffer_size = -1
    if args.buffer.upper() != "AUTO":
        try:
            buffer_size = parse_size(args.buffer)
            if buffer_size <= 0:
                raise ValueError("the value cannot be <= 0")
        except ValueError as e:
            log.error(f"invalid buffer size: {e}")
            return 1

    generator = WordlistGenerator()

    # file mode (-f/--file)
    if args.expr_file is not None:
        if args.start or args.end:
            log.error("--from/--to are not supported with expression files.")
            return 1

        with r_open(args.expr_file, "r", encoding="utf-8") as fp:
            if fp is None:
                return 1

            lines = [line.strip() for line in fp if line.strip()]
            aliases: List[Tuple[str, str]] = []
            current_files: List[str] = []

            log.info(f'Opening file "{args.expr_file}" with {len(lines)} lines.')

            for i, line in enumerate(lines):
                # apply aliases
                for alias_key, alias_val in aliases:
                    line = re.sub(r"(?<!\\)\$" + re.escape(alias_key), alias_val, line)

                fields = line.split(" ")
                keyword = fields[0]

                # apply comments
                if keyword == "#":
                    continue

                # alias definition
                if keyword == r"%alias":
                    if len(fields) < 3:
                        log.error(
                            r"invalid file: '%alias' keyword requires 2 arguments."
                        )
                        return 1
                    aliases.append((fields[1].strip(), " ".join(fields[2:])))
                    continue

                # file include
                if keyword == r"%file":
                    if len(fields) < 2:
                        log.error(
                            r"invalid file: '%file' keyword requires 1 arguments."
                        )
                        return 1
                    current_files.append(" ".join(fields[1:]).strip())
                    continue

                try:
                    tokens = generator.tokenize(line)
                    nodes = generator.parse(tokens, files=(current_files or None))
                    s_bytes, s_words = generator.stats(
                        nodes, sep_len=len(args.separator)
                    )
                    current_files = []  # reset files after usage
                except ExprError as e:
                    log.error(e)
                    return 1

                log.info(
                    f"Generating {s_words:,} words ({format_size(s_bytes)}) for L{i+1}..."
                )

                stats = (s_bytes, s_words)

                ret_code = generate(
                    generator,
                    nodes,
                    stats,
                    filename=args.output,
                    buffering=buffer_size,
                    quiet_mode=args.quiet,
                    sep=args.separator,
                )
                if ret_code != 0:
                    return ret_code
        return 0

    expression, proc_files = f_expression(args.expression, args.files)

    try:
        tokens = generator.tokenize(expression)
        nodes = generator.parse(tokens, files=(proc_files or None))
        s_bytes, s_words = generator.stats(
            nodes, sep_len=len(args.separator), start_from=args.start, end=args.end
        )
    except ExprError as e:
        log.error(e)
        return 1

    log.info(f"Fuse v{__version__}")
    log.info(f"Fuse will generate {s_words:,} words (~{format_size(s_bytes)}).\n")

    if not args.quiet:
        try:
            getpass("Press ENTER to continue...")
        except KeyboardInterrupt:
            sys.stdout.write("\n")
            sys.stdout.flush()
            return 0

    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")
    sys.stdout.flush()

    stats = (s_bytes, s_words)
    wrange = (args.start, args.end)

    try:
        return generate(
            generator,
            nodes,
            stats,
            filename=args.output,
            buffering=buffer_size,
            quiet_mode=args.quiet,
            sep=args.separator,
            wrange=wrange,
        )
    except KeyboardInterrupt:
        log.error("Unexpected keyboard interruption!")

    return 1
