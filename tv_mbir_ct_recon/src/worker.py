from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event
from typing import Callable


@dataclass
class CancellationToken:
    _event: Event = field(default_factory=Event)

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


def run_with_cancellation(fn: Callable[[CancellationToken], object]) -> object:
    token = CancellationToken()
    return fn(token)
