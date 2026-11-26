"""Small function decorators used across the package."""
from __future__ import annotations

import gc
from functools import wraps
from typing import TYPE_CHECKING, ParamSpec, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
R = TypeVar("R")

GC_EVERY_N_TASKS = 10
_tasks_since_gc = 0


def force_gc(func: Callable[P, R]) -> Callable[P, R]:
    """Full garbage collection every GC_EVERY_N_TASKS calls in this process.
    Collecting after every task can cost more than a small model's fit."""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        """Call `func`, then count the call and collect garbage when the count is reached."""
        global _tasks_since_gc
        try:
            return func(*args, **kwargs)
        finally:
            _tasks_since_gc += 1
            if _tasks_since_gc >= GC_EVERY_N_TASKS:
                _tasks_since_gc = 0
                gc.collect(2)

    return wrapper
