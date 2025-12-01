#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys
import multiprocessing
import signal
import ctypes

from types import FrameType
from dataclasses import dataclass
from datetime import datetime
from getpass import getpass
from logging import ERROR
from typing import Any
from time import perf_counter
from pathlib import Path
from fuse import __version__

from fuse.logger import log
from fuse.args import create_parser
from fuse.console import get_progress
from fuse.utils.files import secure_open
from fuse.utils.formatters import format_size, format_time, parse_size
from fuse.utils.generator import ExprError, Node, WordlistGenerator


@dataclass
class GenerateOptions:
    filename: str | None
    buffering: int
    quiet_mode: bool
    separator: str
    wrange: tuple[str | None, str | None]
    filter: str | None
    threads: int


workers: list[multiprocessing.Process] = []


def generate(
    generator: WordlistGenerator,
    nodes: list[Node],
    stats: tuple[int, int],
    options: GenerateOptions,
) -> int:
    global workers

    progress = multiprocessing.Value(ctypes.c_longlong, 0)
    total_bytes, total_words = stats

    event = multiprocessing.Event()
    thread = multiprocessing.Process(
        target=get_progress, args=(event, progress), kwargs={"total": total_bytes}
    )
    show_progress_bar = (options.filename is not None) and (not options.quiet_mode)

    # output file or stdout
    with secure_open(
        options.filename, "a", encoding="utf-8", buffering=options.buffering
    ) as fp:
        if not fp:
            return 1

        start_token, end_token = options.wrange

        if show_progress_bar:
            thread.start()

        start_time = perf_counter()

        # stops progress thread
        def stop_progress() -> None:
            if show_progress_bar and not event.is_set():
                event.set()
                thread.join()

        if options.filter:
            log.warning(
                "Using --filter: Some words may be discarded and performance may be reduced."
            )

        log.info(
            datetime.now().strftime(
                "Wordlist generation started at %H:%M:%S — %a, %b %d %Y."
            )
        )

        try:
            if options.threads > 1:
                # threaded generation
                write_lock = multiprocessing.Lock()

                # calculate indices
                start_idx = 0
                if start_token:
                    start_idx = generator._calculate_skipped_count(nodes, start_token)

                count = total_words
                step = count // options.threads
                remainder = count % options.threads

                current_idx = start_idx

                def worker(
                    w_start: str, w_end: str | None, p_val: Any, lock: Any
                ) -> None:
                    buf = []
                    buf_size = 1000

                    try:
                        with secure_open(
                            options.filename,
                            "a",
                            encoding="utf-8",
                            buffering=options.buffering,
                        ) as fp_worker:
                            if not fp_worker:
                                return

                            for token in generator.generate(
                                nodes, start_from=w_start, end=w_end
                            ):
                                if options.filter is not None and not re.match(
                                    options.filter, token
                                ):
                                    with lock:
                                        p_val.value += len(token + options.separator)
                                    continue

                                buf.append(token + options.separator)
                                if len(buf) >= buf_size:
                                    data = "".join(buf)
                                    with lock:
                                        fp_worker.write(data)
                                        p_val.value += len(data)
                                    buf.clear()

                            if buf:
                                data = "".join(buf)
                                with lock:
                                    fp_worker.write(data)
                                    p_val.value += len(data)

                    except Exception as e:
                        log.error(f"worker error: {e}")

                def workers_shutdown(signum: int, frame: FrameType | None) -> None:
                    stop_progress()
                    
                    for worker in workers:
                        if worker.is_alive():
                            worker.terminate()

                    for worker in workers:
                        worker.join()

                    log.warning("Generation stopped with keyboard interrupt!")
                    
                    sys.exit(0)

                signal.signal(signal.SIGINT, workers_shutdown)

                for i in range(options.threads):
                    t_count = step + (1 if i < remainder else 0)
                    if t_count == 0:
                        continue

                    if i == 0:
                        w_start = start_token
                    else:
                        w_start = generator.get_word_at_index(nodes, current_idx - 1)

                    current_idx += t_count
                    w_end = generator.get_word_at_index(nodes, current_idx - 1)

                    p = multiprocessing.Process(
                        target=worker, args=(w_start, w_end, progress, write_lock)
                    )
                    workers.append(p)
                    p.start()

                for p in workers:
                    p.join()

            else:
                try:
                    for token in generator.generate(nodes, start_from=start_token):
                        if options.filter is not None and not re.match(
                            options.filter, token
                        ):
                            progress.value += len(token + options.separator)
                            continue

                        progress.value += fp.write(token + options.separator)

                        if end_token == token:
                            stop_progress()
                            break
                except KeyboardInterrupt:
                    stop_progress()
                    log.warning("Generation stopped with keyboard interrupt!")

                    return 1
        except re.PatternError as err:
            stop_progress()
            log.error(f"invalid filter: {err}.")

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


def format_expression(expression: str, files: list[str]) -> tuple[str, list[str]]:
    n_files = 0
    files_out: list[str] = []

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

    if args.workers > 50 or args.workers < 1:
        log.error(f"invalid number of workers ({args.workers}). choose a value between 1 and 50.")
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

    gen_options = GenerateOptions(
        filename=args.output,
        buffering=buffer_size,
        quiet_mode=args.quiet,
        separator=args.separator,
        wrange=(args.start, args.end),
        filter=args.filter,
        threads=args.workers,
    )

    generator = WordlistGenerator()

    # file mode (-f/--file)
    if args.expr_file is not None:
        if args.start or args.end:
            log.error("--from/--to are not supported with expression files.")
            return 1

        with secure_open(args.expr_file, "r", encoding="utf-8") as fp:
            if fp is None:
                return 1

            lines = [line.strip() for line in fp if line.strip()]
            aliases: list[tuple[str, str]] = []
            current_files: list[str] = []

            log.info(f'Opening file "{args.expr_file}" (with {len(lines)} lines).')

            for i, line in enumerate(lines):
                # apply aliases
                for alias_key, alias_val in aliases:
                    line = re.sub(
                        r"(?<!\\)\$" + re.escape(alias_key) + ";", alias_val, line
                    )

                fields = line.split(" ")
                keyword = fields[0]
                arguments = fields[1:]

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
                    a_name = arguments[0].strip()
                    a_value = " ".join(arguments[1:])
                    if ";" in a_name:
                        log.error(r"invalid file: '%alias' cannot contain ';'.")
                        return 1
                    aliases.append((a_name, a_value))
                    continue

                # file include
                if keyword == r"%file":
                    if len(fields) < 2:
                        log.error(
                            r"invalid file: '%file' keyword requires 1 arguments."
                        )
                        return 1

                    if arguments[0].startswith("./"):
                        # get abs path
                        base_dir = Path(Path(args.expr_file).resolve()).parent
                        file = str((base_dir / " ".join(arguments).strip()).resolve())
                    else:
                        file = " ".join(arguments).strip()

                    current_files.append(file)
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

                ret_code = generate(generator, nodes, stats, gen_options)
                if ret_code != 0:
                    return ret_code
        return 0

    expression, proc_files = format_expression(args.expression, args.files)

    try:
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
    except OverflowError:
        log.error("Overflow Error! Is the expression correct?")
        return 1

    if not args.quiet:
        try:
            getpass("Press ENTER to continue...")
            sys.stdout.write("\033[F")
            sys.stdout.write("\033[K")
            sys.stdout.flush()
        except KeyboardInterrupt:
            sys.stdout.write("\n")
            sys.stdout.flush()
            return 0

    stats = (s_bytes, s_words)

    try:
        return generate(generator, nodes, stats, gen_options)
    except KeyboardInterrupt:
        log.error("Unexpected keyboard interruption!")
    finally:
        sys.stdout.write("\033[?25h")  # fix cursor bug

    return 1
