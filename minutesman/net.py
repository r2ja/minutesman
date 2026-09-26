# OpenAI client that survives long requests: TCP keepalive plus a "no data" read timeout instead of a total timeout
from __future__ import annotations

import logging
import os
import socket

from openai import OpenAI

log = logging.getLogger(__name__)
KEEPALIVE_IDLE = 30  # seconds of silence before the first keepalive probe
KEEPALIVE_INTERVAL = 15
KEEPALIVE_COUNT = 8
READ_STALL_SECONDS = 240  # a stream that sends nothing for this long is treated as dead


# Keepalive options this OS accepts (checked on a throwaway socket so connects never fail)
def keepalive_options() -> list[tuple[int, int, int]]:
    wanted = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    for name, value in (("TCP_KEEPIDLE", KEEPALIVE_IDLE), ("TCP_KEEPALIVE", KEEPALIVE_IDLE),
                        ("TCP_KEEPINTVL", KEEPALIVE_INTERVAL), ("TCP_KEEPCNT", KEEPALIVE_COUNT)):
        if hasattr(socket, name):
            wanted.append((socket.IPPROTO_TCP, getattr(socket, name), value))
    ok = []
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        for opt in wanted:
            try:
                probe.setsockopt(*opt)
                ok.append(opt)
            except OSError:
                pass
    return ok


def make_client(max_retries: int = 5) -> OpenAI:
    try:
        import httpx2 as hx
    except ImportError:
        import httpx as hx
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    transport = hx.HTTPTransport(socket_options=keepalive_options(), proxy=proxy)
    timeout = hx.Timeout(connect=30.0, read=READ_STALL_SECONDS, write=300.0, pool=60.0)
    return OpenAI(http_client=hx.Client(transport=transport, timeout=timeout), timeout=timeout,
                  max_retries=max_retries)
