from rich.console import Console
from rich.text import Text
from io import StringIO


class ExprError(Exception):
    def __init__(self, message: str, error_pos: tuple[str, int] | None = None) -> None:
        if error_pos is not None:
            pattern, char_pos = error_pos

            window = 24

            if pattern:
                char_pos = max(0, min(char_pos - 1, len(pattern) - 1))

                start = max(0, char_pos - window // 2)
                end = min(len(pattern), start + window)
                start = max(0, end - window)

                snippet = Text()

                if start > 0:
                    snippet.append("...", style="magenta dim")

                for i, ch in enumerate(pattern[start:end]):
                    if start + i == char_pos:
                        snippet.append(ch, style="on red")
                    else:
                        snippet.append(ch)

                if end < len(pattern):
                    snippet.append("...", style="magenta dim")

                caret_pos = (char_pos - start) + (3 if start > 0 else 0)
            else:
                snippet = Text("")
                caret_pos = 0

            rich_buffer = StringIO()
            console = Console(file=rich_buffer, highlight=True, force_terminal=True)

            console.print(snippet)
            console.print(" " * max(0, caret_pos - 1) + "[bold blue]^^^[/]", end="")

            message += "\n\n" + rich_buffer.getvalue()

        super().__init__("expression error: " + message)
