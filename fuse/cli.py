#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import sys
import tty
import termios
import signal
import ctypes
import multiprocessing

from types import FrameType
from dataclasses import dataclass
from datetime import datetime
from logging import ERROR
from typing import Any
from time import perf_counter
from fuse import __version__

from fuse.logger import log
from fuse.args import create_parser
from fuse.console import get_progress
from fuse.files import secure_open
from fuse.core.formatters import format_size, format_time, parse_size
from fuse.core.generator import ExprError, Node, WordlistGenerator
from fuse.file_parser import InvalidSyntaxError, process_expr_file


@dataclass
class GenerateOptions:
    filename: str | None
    buffering: int
    quiet_mode: bool
    separator: str
    wrange: tuple[str | None, str | None]
    filter: str | None
    threads: int


def generate(
    generator: WordlistGenerator,
    nodes: list[Node],
    stats: tuple[int, int],
    options: GenerateOptions,
) -> int:
    workers: list[multiprocessing.Process] = []
    progress = multiprocessing.Value(ctypes.c_longlong, 0)
    total_bytes, total_words = stats

    pattern = None
    if options.filter is not None:
        try:
            pattern = re.compile(options.filter)
        except re.PatternError as err:
            log.error(f"invalid filter: {err}.")
            return 1

        log.warning("Filtering can discard words and reduce performance.")

    if options.threads > 1:
        log.warning(f"Using multiple workers may result in interleaved output.")

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

        log.info(
            datetime.now().strftime(
                "Wordlist generation started at %H:%M:%S on %a, %b %d %Y."
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
                                if pattern is not None and not re.match(pattern, token):
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

                    log.error("Generation stopped with keyboard interrupt!")

                    sys.exit(1)

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
                        if pattern is not None and not re.match(pattern, token):
                            progress.value += len(token + options.separator)
                            continue

                        progress.value += fp.write(token + options.separator)

                        if end_token == token:
                            stop_progress()
                            break
                except KeyboardInterrupt:
                    stop_progress()
                    log.error("Generation stopped with keyboard interrupt!")

                    return 1
        except Exception:
            stop_progress()
            raise

        elapsed = perf_counter() - start_time
        stop_progress()

    if show_progress_bar and thread.is_alive():
        thread.join()

    speed = int(total_words / elapsed) if elapsed > 0 else 0
    log.info(f"Finished in {format_time(elapsed)} ({speed} W/s).")

    return 0


def pause(prompt: str = "Press the Enter key to continue") -> bool:
    if not sys.stdin.isatty():
        return True

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        sys.stderr.write(prompt)
        sys.stderr.flush()

        tty.setraw(fd)

        while True:
            ch = sys.stdin.read(1)
            if ch in ("\r", "\n"):
                break
            elif ch == "\x03":
                raise KeyboardInterrupt

        sys.stderr.write("\r\033[K")
        sys.stderr.flush()

        return True

    except KeyboardInterrupt:
        sys.stderr.write("\r\n")
        sys.stderr.flush()
        return False

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def format_expression(expression: str, files: list[str]) -> tuple[str, list[str]]:
    files_out: list[str] = []

    for file_path in files:
        if file_path.startswith("//"):
            inline = file_path.replace("//", "", 1)
            expression = re.sub(r"(?<!\\)\^", lambda m: inline, expression, count=1)
        else:
            files_out.append(file_path)

    return expression, files_out


def main() -> int:
    parser = create_parser()
    args = parser.parse_args()

    if args.expression is None and args.expr_file is None:
        parser.print_help(sys.stderr)
        return 1

    if not (1 <= args.workers <= 64):
        log.error(
            f"invalid number of workers ({args.workers}). choose a value between 1 and 64."
        )
        return 1

    if args.quiet:
        log.setLevel(ERROR)

    buffer_size = args.buffer

    if isinstance(buffer_size, str):
        try:
            buffer_size = parse_size(buffer_size)
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

        try:
            for i, (expression, expr_files) in enumerate(
                process_expr_file(args.expr_file)
            ):
                try:
                    tokens = generator.tokenize(expression)
                    nodes = generator.parse(tokens, files=(expr_files or None))
                    s_bytes, s_words = generator.stats(
                        nodes, sep_len=len(args.separator)
                    )
                except ExprError as e:
                    log.error(e)
                    return 1

                log.info(
                    f"Generating {s_words} words ({format_size(s_bytes)}) for L{i+1}..."
                )

                stats = (s_bytes, s_words)

                ret_code = generate(generator, nodes, stats, gen_options)
                if ret_code != 0:
                    return ret_code
        except InvalidSyntaxError as e:
            log.error(e)
            return 1

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
        log.info(f"Fuse will generate {s_words} words (~{format_size(s_bytes)}).\n")
    except (OverflowError, ValueError):
        log.error("Overflow Error! Is the expression correct?")
        return 1

    if not (args.quiet or args.non_interactive) and not pause():
        return 1

    stats = (s_bytes, s_words)

    try:
        return generate(generator, nodes, stats, gen_options)
    except KeyboardInterrupt:
        log.error("Unexpected keyboard interruption!")
    finally:
        sys.stdout.write("\033[?25h")  # fix cursor bug

    return 1
