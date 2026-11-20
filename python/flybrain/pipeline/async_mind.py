"""Asynchronous mind: one background thought at a time; the motor never waits.

The live loop snapshots the senses and submits a thought job whenever the worker
is idle; the motor keeps tracking the last committed directive every step. A new
stimulus (stimulus interrupt) invalidates the in-flight thought so a plan made
for the old world is never committed — the fly pauses and deep-thinks on the new
stimulus instead (reframe).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures
from dataclasses import dataclass
from typing import Any

MIND_JOB_THINK = "think"
MIND_JOB_DEEP = "deep"


@dataclass(frozen=True)
class MindJobResult:
    kind: str
    value: Any
    submitted_step: int


class AsyncMindScheduler:
    """Holds at most one in-flight mind job; stale results are discarded on poll."""

    def __init__(self, executor: Executor | None = None) -> None:
        self._executor = executor
        self._owns_executor = executor is None
        self._future: Future | None = None
        self._kind = ""
        self._submitted_step = 0
        self._job_generation = 0
        self.generation = 0
        self.submitted = 0
        self.committed = 0
        self.discarded = 0
        self.last_discard_reason = ""

    @property
    def busy(self) -> bool:
        return self._future is not None

    @property
    def inflight_kind(self) -> str | None:
        return self._kind if self._future is not None else None

    def submit(self, kind: str, fn: Callable[[], Any], *, step: int) -> bool:
        if self.busy:
            return False
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fly-mind")
        self._kind = kind
        self._submitted_step = int(step)
        self._job_generation = self.generation
        self.submitted += 1
        self._future = self._executor.submit(fn)
        return True

    def invalidate(self, reason: str) -> bool:
        """Discard the in-flight job's result when it finishes. True if one was in flight."""
        self.generation += 1
        if self._future is None:
            return False
        self.last_discard_reason = reason
        return True

    def poll(self) -> MindJobResult | None:
        """Finished, still-current job result (worker errors re-raise here), else None."""
        future = self._future
        if future is None or not future.done():
            return None
        self._future = None
        value = future.result()
        if self._job_generation != self.generation:
            self.discarded += 1
            return None
        self.committed += 1
        return MindJobResult(kind=self._kind, value=value, submitted_step=self._submitted_step)

    def wait(self, timeout: float | None = None) -> None:
        if self._future is not None:
            wait_futures([self._future], timeout=timeout)

    def shutdown(self) -> None:
        self.wait()
        self._future = None
        if self._owns_executor and self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def stats(self) -> dict[str, Any]:
        return {
            "inflight": self.inflight_kind,
            "submitted": self.submitted,
            "committed": self.committed,
            "discarded": self.discarded,
            "last_discard_reason": self.last_discard_reason,
        }


class FairLock:
    """FIFO mutex: the mind worker gets the next turn instead of starving behind the
    motor loop, which re-acquires immediately after each short step."""

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._next_ticket = 0
        self._serving = 0

    def acquire(self) -> bool:
        with self._cond:
            ticket = self._next_ticket
            self._next_ticket += 1
            while ticket != self._serving:
                self._cond.wait()
        return True

    def release(self) -> None:
        with self._cond:
            self._serving += 1
            self._cond.notify_all()

    def __enter__(self) -> FairLock:
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class SerializedAccess:
    """Proxy serializing calls on a shared backend (e.g. STM) between mind and motor threads."""

    def __init__(self, target: Any, lock: threading.RLock | None = None) -> None:
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_lock", lock or threading.RLock())

    @property
    def target(self) -> Any:
        return self._target

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        if not callable(attr):
            return attr
        lock = self._lock

        def call(*args: Any, **kwargs: Any) -> Any:
            with lock:
                return attr(*args, **kwargs)

        return call

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._target, name, value)
