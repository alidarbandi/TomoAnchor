from __future__ import annotations

from typing import Callable


class OperationCancelled(RuntimeError):
    """Raised when a user-triggered cancellation should stop the current task."""


def raise_if_cancelled(cancel_check: Callable[[], bool] | None, message: str = "Operation cancelled.") -> None:
    if cancel_check is not None and cancel_check():
        raise OperationCancelled(message)
