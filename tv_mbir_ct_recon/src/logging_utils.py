from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

from .io_utils import replace_file_atomically, temporary_output_path


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class MemoryLog:
    def __init__(
        self,
        callback: Callable[[str], None] | None = None,
        verbose: bool = True,
        persist_path: str | Path | None = None,
    ) -> None:
        self.lines: list[str] = []
        self.callback = callback
        self.verbose = verbose
        self.persist_path = None if persist_path is None else Path(persist_path)

    def write(self, message: str) -> None:
        line = message if message.startswith("[") else f"[{timestamp()}] {message}"
        self.lines.append(line)
        if self.persist_path is not None:
            write_text_lines(self.persist_path, [line], append=True)
        if self.callback is not None:
            self.callback(line)
        elif self.verbose:
            print(line)

    def extend(self, messages: Iterable[str]) -> None:
        for message in messages:
            self.write(message)

    def save(self, path: str | Path) -> Path:
        return write_text_lines(path, self.lines, append=False)


def append_text_line(path: str | Path, line: str) -> Path:
    return write_text_lines(path, [line], append=True)


def write_text_lines(path: str | Path, lines: Iterable[str], append: bool = False) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(line) for line in lines) + "\n"
    if append:
        with output.open("a", encoding="utf-8") as handle:
            handle.write(text)
        return output
    temp = temporary_output_path(output, suffix=".txt.tmp")
    with temp.open("w", encoding="utf-8") as handle:
        handle.write(text)
    return replace_file_atomically(temp, output)
