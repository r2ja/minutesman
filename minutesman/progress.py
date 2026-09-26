# Progress logging: heartbeats while waiting on slow API calls, counters for parallel stages
from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager

log = logging.getLogger("minutesman")
HEARTBEAT_SECONDS = 30


# Log "still waiting" every HEARTBEAT_SECONDS until the block finishes
@contextmanager
def heartbeat(label: str, every: float = HEARTBEAT_SECONDS):
    start = time.time()
    done = threading.Event()

    def beat():
        while not done.wait(every):
            log.info("  ... still waiting on %s (%s)", label, fmt(time.time() - start))

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
