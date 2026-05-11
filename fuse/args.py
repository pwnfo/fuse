from fuse import __description__, __author__, __credits__, __version__, BANNER

import sys
import argparse

from fuse.logger import log
from typing import Never


class FuseParser(argparse.ArgumentParser):
    """Format `argparse.ArgumentParser` error message"""

    def error(self, message: str) -> Never:
        self.print_usage(sys.stderr)
        log.error("\n" + message)
        sys.exit(1)


def create_parser(prog: str = "fuse") -> FuseParser:
    """Create the main CLI argument parser"""
    parser = FuseParser(
        prog=prog,
        add_help=False,
        usage=f"{prog} [options] <expression> [<files...>]",
        description=__description__,
        epilog=__credits__
        + "\nMore information and examples:\n  https://fuse-generator.readthedocs.io/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # argument groups
    general = parser.add_argument_group("General Options")
    generation = parser.add_argument_group("Generation Options")

    general.add_argument(
        "-h", "--help", action="help", help="show this help message and exit"
    )
    general.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"Fuse v{__version__} (Python {sys.version_info.major}.{sys.version_info.minor})",
        help="show version message and exit",
    )
    general.add_argument(
        "-o",
        "--output",
        metavar="<path>",
        dest="output",
        help="write the wordlist in the file",
    )
    general.add_argument(
        "-f",
        "--file",
        metavar="<path>",
        dest="expr_file",
        help="file with different expressions",
    )
    general.add_argument(
        "-q", "--quiet", action="store_true", dest="quiet", help="use quiet mode"
    )
    general.add_argument(
        "-n",
        "--non-interactive",
        action="store_true",
        dest="non_interactive",
        help="disable interactive prompt before execution",
    )

    generation.add_argument(
        "-d",
        "--delimiter",
        metavar="<string>",
        dest="delimiter",
        default="\n",
        help="delimiter between entries",
    )
    generation.add_argument(
        "-b",
        "--write-buffer",
        metavar="<size>",
        dest="buffer",
        default=-1,
        help="write buffer size",
    )
    generation.add_argument(
        "-w",
        "--workers",
        metavar="<1-64>",
        dest="workers",
        type=int,
        default=1,
        help="number of workers (default is 1)",
    )
    generation.add_argument(
        "-F",
        "--filter",
        metavar="<regex>",
        dest="filter",
        help="filter generated words using a regex",
    )
    generation.add_argument(
        "-k",
        "--flush-threshold",
        metavar="<size>",
        dest="flush",
        help="byte threshold before flushing output (default is 512KB)",
        default="512KB",
    )
    generation.add_argument(
        "-z",
        "--compress",
        metavar="<format>",
        dest="compress",
        choices=["gzip", "bzip2", "lzma"],
        help="compress output (available: gzip, bzip2 and lzma)",
    )
    generation.add_argument(
        "-l",
        "--compresslevel",
        metavar="<level>",
        type=int,
        dest="compresslevel",
        help="compression level (depends on selected format)",
    )
    generation.add_argument(
        "-S",
        "--start",
        metavar="<word>",
        dest="start",
        help="start writing the wordlist from <word>",
    )
    generation.add_argument(
        "-E",
        "--end",
        metavar="<word>",
        dest="end",
        help="end writing the wordlist at <word>",
    )

    # positional arguments
    parser.add_argument("expression", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("files", nargs="*", help=argparse.SUPPRESS)

    return parser
