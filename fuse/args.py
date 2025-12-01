from fuse import __description__, __author__, __version__

import sys
import argparse

from fuse.logger import log

from typing import Never


class FuseParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        self.print_usage(sys.stderr)
        sys.stderr.write("\n")
        log.error(message)
        sys.exit(1)


def create_parser(prog: str = "fuse") -> FuseParser:
    parser = FuseParser(
        prog=prog,
        add_help=False,
        usage=f"{prog} [options] <expression> [<files...>]",
        description=f"Fuse v{__version__}",
        epilog=__description__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    options = parser.add_argument_group()

    # fmt: off
    options.add_argument("-h", "--help", action="help", help="show this help message and exit")
    options.add_argument("-v", "--version", action="version", version=f"Fuse v{__version__} (Python {sys.version_info.major}.{sys.version_info.minor})", help="show version message and exit")
    options.add_argument("-o", "--output", metavar="+", dest="output", help="write the wordlist in the file")
    options.add_argument("-f", "--file", metavar="+", dest="expr_file", help="files with different expressions")
    options.add_argument("-q", "--quiet", action="store_true", dest="quiet", help="use quiet mode")
    options.add_argument("-s", "--separator", metavar="+", dest="separator", default="\n", help="separator between entries")
    options.add_argument("-b", "--buffer", metavar="+", dest="buffer", default="AUTO", help="buffer size in wordlist generation")
    options.add_argument("-w", "--workers", metavar="+", dest="workers", type=int, default=2, help="number of workers (default is 2)")
    options.add_argument("-F", "--filter", metavar="+", dest="filter", help="filter generated words using a regex")
    options.add_argument("-S", "--from", metavar="+", dest="start", help="start writing the wordlist with <word>")
    options.add_argument("-E", "--to", metavar="+", dest="end", help="ends writing the wordlist with <word>")

    parser.add_argument("expression", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("files", nargs="*", help=argparse.SUPPRESS)
    # fmt: on

    return parser
