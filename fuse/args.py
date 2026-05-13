from fuse import __description__, __author__, __credits__, __version__, BANNER

import sys
import argparse

from fuse.logger import log
from typing import Never

CLI_EXAMPLES = """
Examples:
  fuse '/A{4}'
  fuse '/l{4}#[0-8:2]' -o words.txt
  fuse -f patterns.fuse
  fuse '/h{6}' -F '.{4}ff' -w 3 -n
  fuse '/H{6}' -z gzip -k 1MB -l 7 -o hashes.txt.gz
"""
class FuseParser(argparse.ArgumentParser):
    """Format `argparse.ArgumentParser` error message"""

    def error(self, message: str) -> Never:
        self.print_usage(sys.stderr)
        log.error("\n" + message)
        sys.exit(1)


def create_parser(prog: str = "fuse") -> FuseParser:
    """Create the main CLI argument parser"""
    epilog = CLI_EXAMPLES + "\n" + __credits__ + "\nMore information and examples:\n  https://fuse-generator.readthedocs.io/"
    parser = FuseParser(
        prog=prog,
        add_help=False,
        usage=f"{prog} [options] <expression> [<files...>]",
        description=__description__,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # argument groups
    general_group = parser.add_argument_group("General Options")
    generation_group = parser.add_argument_group("Generation Options")
    input_group = parser.add_argument_group("Input Options")
    output_group = parser.add_argument_group("Output Options")

    general_group.add_argument(
        "-h", "--help", action="help", help="show this help message and exit"
    )
    general_group.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"Fuse v{__version__} (Python {sys.version_info.major}.{sys.version_info.minor})",
        help="show version information and exit",
    )
    general_group.add_argument(
        "-q", "--quiet", action="store_true", dest="quiet", help="suppress non-essential output"
    )
    general_group.add_argument(
        "-n",
        "--non-interactive",
        action="store_true",
        dest="non_interactive",
        help="skip the confirmation prompt before execution",
    )

    input_group.add_argument(
        "-f",
        "--file",
        metavar="<path>",
        dest="expr_file",
        help="load expressions from file",
    )
    input_group.add_argument(
        "-S",
        "--start",
        metavar="<word>",
        dest="start",
        help="start writing output from <word>",
    )
    input_group.add_argument(
        "-E",
        "--end",
        metavar="<word>",
        dest="end",
        help="stop writing output at <word>",
    )


    generation_group.add_argument(
        "-d",
        "--delimiter",
        metavar="<string>",
        dest="delimiter",
        default="\n",
        help="string inserted between generated entries",
    )
    generation_group.add_argument(
        "-b",
        "--write-buffer",
        metavar="<size>",
        dest="buffer",
        default=-1,
        help="output buffer size",
    )
    generation_group.add_argument(
        "-w",
        "--workers",
        metavar="<1-64>",
        dest="workers",
        type=int,
        default=1,
        help="number of worker processes (default: 1)",
    )
    generation_group.add_argument(
        "-F",
        "--filter",
        metavar="<regex>",
        dest="filter",
        help="filter generated words with a regular expression",
    )
    generation_group.add_argument(
        "-k",
        "--flush-threshold",
        metavar="<size>",
        dest="flush",
        help="flush output after reaching this byte threshold (default: 512KB)",
        default="512KB",
    )

    output_group.add_argument(
        "-o",
        "--output",
        metavar="<path>",
        dest="output",
        help="write output to a file",
    )
    output_group.add_argument(
        "-z",
        "--compress",
        metavar="<format>",
        dest="compress",
        choices=["gzip", "bzip2", "lzma"],
        help="compress output (supported: gzip, bzip2, lzma)",
    )
    output_group.add_argument(
        "-l",
        "--compresslevel",
        metavar="<level>",
        type=int,
        dest="compresslevel",
        help=" compression level for the selected format",
    )
    
    
    # positional arguments
    parser.add_argument("expression", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("files", nargs="*", help=argparse.SUPPRESS)

    return parser
