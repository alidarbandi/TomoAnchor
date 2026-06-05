from __future__ import annotations

from src.logging_utils import MemoryLog, append_text_line, write_text_lines


def test_write_text_lines_append_mode(tmp_path) -> None:
    path = tmp_path / "log.txt"

    write_text_lines(path, ["first"])
    write_text_lines(path, ["second", "third"], append=True)

    assert path.read_text(encoding="utf-8") == "first\nsecond\nthird\n"


def test_append_text_line_adds_single_line(tmp_path) -> None:
    path = tmp_path / "log.txt"

    append_text_line(path, "alpha")
    append_text_line(path, "beta")

    assert path.read_text(encoding="utf-8") == "alpha\nbeta\n"


def test_memory_log_persists_lines_while_running(tmp_path) -> None:
    path = tmp_path / "log.txt"
    seen: list[str] = []
    log = MemoryLog(callback=seen.append, verbose=False, persist_path=path)

    log.write("worker started")
    log.write("[2026-06-04 10:00:00] already stamped")

    text = path.read_text(encoding="utf-8")
    assert "worker started" in text
    assert "[2026-06-04 10:00:00] already stamped" in text
    assert len(seen) == 2
