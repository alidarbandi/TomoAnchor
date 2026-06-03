from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class MemoryLog:
    def __init__(self, callback: Callable[[str], None] | None = None, verbose: bool = True) -> None:
        self.lines: list[str] = []
        self.callback = callback
        self.verbose = verbose

    def write(self, message: str) -> None:
        line = message if message.startswith("[") else f"[{timestamp()}] {message}"
        self.lines.append(line)
        if self.callback is not None:
            self.callback(line)
        elif self.verbose:
            print(line)

    def extend(self, messages: Iterable[str]) -> None:
        for message in messages:
            self.write(message)

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        return output


def write_text_lines(path: str | Path, lines: Iterable[str]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
