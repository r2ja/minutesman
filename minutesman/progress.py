# Progress logging: heartbeats while waiting on slow API calls, counters for parallel stages
from __future__ import annotations

import logging
import queue
import threading
import time
from contextlib import contextmanager

log = logging.getLogger("minutesman")
HEARTBEAT_SECONDS = 30


# Log "still waiting" every HEARTBEAT_SECONDS until the block finishes; status() adds live detail
@contextmanager
def heartbeat(label: str, every: float = HEARTBEAT_SECONDS, status=None):
    start = time.time()
    done = threading.Event()

    def beat():
        while not done.wait(every):
            extra = f", {status()}" if status else ""
            log.info("  ... still waiting on %s (%s%s)", label, fmt(time.time() - start), extra)

    t = threading.Thread(target=beat, daemon=True)
    t.start()
    try:
        yield
    finally:
        done.set()


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


# Thread-safe "n/total done" counter that logs about every 10% and at the end
class Counter:
    def __init__(self, label: str, total: int):
        self.label, self.total, self.n = label, total, 0
        self.start = time.time()
        self.step = max(1, total // 10)
        self.lock = threading.Lock()

    def tick(self) -> None:
        with self.lock:
            self.n += 1
            if self.n % self.step == 0 or self.n == self.total:
                log.info("%s: %d/%d done (%s)", self.label, self.n, self.total, fmt(time.time() - self.start))


# Run call(); if it hasn't returned after hedge_after seconds, start a second copy and take whichever finishes first
def hedged(call, hedge_after: float, label: str):
    results: queue.Queue = queue.Queue()

    def attempt(n):
        try:
            results.put((n, None, call()))
        except BaseException as exc:  # noqa: BLE001
            results.put((n, exc, None))

    threading.Thread(target=attempt, args=(1,), daemon=True).start()
    running, error = 1, None
    try:
        n, error, value = results.get(timeout=hedge_after)
        if error is None:
            return value
        running = 0
        log.info("%s failed (%s); trying once more", label, error)
    except queue.Empty:
        log.info("%s is slow; sending a second copy, the first to finish wins", label)
    threading.Thread(target=attempt, args=(2,), daemon=True).start()
    running += 1
    while running:
        n, err, value = results.get()
        running -= 1
        if err is None:
            if n == 2:
                log.info("%s: the second copy finished first", label)
            return value
        error = err
    raise error
