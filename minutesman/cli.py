# Command line: run, estimate, check, shrink
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path

from . import __version__, audio, voiceprint
from .config import LLM_PRICE_PER_1M, TRANSCRIBE_PRICE_PER_MIN, Settings


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path.cwd() / ".env")


def _voices(items: list[str]) -> dict[str, Path]:
    out = {}
    for item in items or []:
        name, sep, path = item.partition("=")
        if not sep or not Path(path).exists():
            raise SystemExit(f"--voice expects NAME=path/to/sample.wav (got {item!r})")
        out[name.strip()] = Path(path)
    return out


# LLM tokens per audio minute, measured with gpt-6-sol at medium effort (non-stop speech, so an upper bound)
TOKENS_IN_PER_MIN = 1600
TOKENS_OUT_PER_MIN = 1150  # includes reasoning tokens
TOKENS_IN_PER_CHUNK = 1000  # system prompt + context, once per chunk
EFFORT_OUTPUT_FACTOR = {"none": 0.6, "low": 0.75, "medium": 1.0, "high": 1.6, "xhigh": 2.4}


# Projected cost from the measured token rates above
def _seconds(text: str) -> float:
    total = 0.0
    for part in text.split(":"):
        total = total * 60 + float(part)
    return total


def _renames(text: str | None) -> dict | None:
    if not text:
        return None
    out = {}
    for part in text.split(","):
        sid, sep, name = part.partition("=")
        if not sep or not name.strip():
            raise SystemExit(f"--rename expects ID=Name pairs like S2=Raja (got {part!r})")
        out[sid.strip()] = name.strip()
    return out


def estimate(minutes: float, cfg: Settings) -> dict:
    a = minutes * (1 + cfg.overlap_seconds / cfg.chunk_seconds) * TRANSCRIBE_PRICE_PER_MIN[cfg.diarize_model]
    b = minutes * 1.01 * TRANSCRIBE_PRICE_PER_MIN.get(cfg.transcribe_model, 0.006)
    chunks = max(1, round(minutes / (cfg.chunk_seconds / 60)))
    tok_in = minutes * TOKENS_IN_PER_MIN + (chunks + 1) * TOKENS_IN_PER_CHUNK
    tok_out = minutes * TOKENS_OUT_PER_MIN * EFFORT_OUTPUT_FACTOR.get(cfg.reasoning_effort, 1.0)
    p_in, _, p_out = LLM_PRICE_PER_1M.get(cfg.llm_model, (2.0, 0.2, 10.0))
    llm_usd = (tok_in * p_in + tok_out * p_out) / 1e6
    return {"pass_a": round(a, 3), "pass_b": round(b, 3), "llm": round(llm_usd, 3),
            "total": round(a + b + llm_usd, 3)}


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    cfg = Settings()
    ap = argparse.ArgumentParser(prog="minutesman", description="Diarized Urdu/English meeting transcripts in Roman Urdu")
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="transcribe + diarize a recording")
    r.add_argument("audio", help="audio/video file, or a share link (Google Drive, Dropbox, direct URL)")
    r.add_argument("-o", "--out", type=Path, help="output folder (default: output/<file name>)")
    r.add_argument("--voice", action="append", metavar="NAME=FILE",
                   help="voice sample of a known participant (5-10 s of them alone); repeatable")
    r.add_argument("--context", help="one line about the meeting, e.g. 'Weekly sync, Acme Karachi'")
    r.add_argument("--keywords", help="comma-separated names/terms, e.g. 'Ahmed,Sara,Jira,Q3'")
    r.add_argument("--llm", dest="llm_model", help=f"fusion LLM (default {cfg.llm_model})")
    r.add_argument("--effort", dest="reasoning_effort", choices=["none", "low", "medium", "high", "xhigh"])
    r.add_argument("--enhance", choices=sorted(audio.ENHANCE_FILTERS))
    r.add_argument("--chunk-seconds", type=int)
    r.add_argument("--no-voiceprints", action="store_true")
    r.add_argument("--name-threshold", type=float)
    r.add_argument("--fresh", action="store_true", help="ignore cached API results")
    r.add_argument("--rename", help='fix names by speaker id, e.g. "S2=Raja,S9=Hamza" (ids are in the speaker table)')
    r.add_argument("--start", help="process from this time, e.g. 2:30")
    r.add_argument("--end", help="stop at this time, e.g. 1:52:00 (if recording was left running)")
    r.add_argument("--min-speaker-seconds", type=int, help="fold speakers with less speech than this (default 15)")

    e = sub.add_parser("estimate", help="projected API cost for a file or a duration")
    e.add_argument("target", help="audio file or minutes (e.g. 80)")
    e.add_argument("--llm", dest="llm_model")
    e.add_argument("--effort", dest="reasoning_effort")

    sub.add_parser("check", help="verify ffmpeg, API key, model access, voiceprint extra")

    sh = sub.add_parser("shrink", help="make a small speech-quality copy (Opus) for uploading")
    sh.add_argument("audio", type=Path)
    sh.add_argument("-o", "--out", type=Path, help="default: <name>.small.ogg next to the input")
    sh.add_argument("--bitrate", default="32k")

    cl = sub.add_parser("clip", help="cut a voice sample for --voice, e.g. clip rec.m4a 0:54 1:04 -o voices/raja.wav")
    cl.add_argument("audio", type=Path)
    cl.add_argument("start", help="mm:ss or h:mm:ss")
    cl.add_argument("end", help="mm:ss or h:mm:ss")
    cl.add_argument("-o", "--out", type=Path, required=True)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    from .progress import quiet_libraries

    quiet_libraries()

    if args.cmd == "estimate":
        cfg.update(llm_model=args.llm_model, reasoning_effort=args.reasoning_effort)
        t = Path(args.target)
        minutes = audio.duration(t) / 60 if t.exists() else float(args.target)
        est = estimate(minutes, cfg)
        print(f"{minutes:.1f} min with {cfg.llm_model} (effort {cfg.reasoning_effort}):")
        for k, v in est.items():
            print(f"  {k:7s} ${v:.3f}")
        return 0

    if args.cmd == "check":
        return check(cfg)

    if args.cmd == "clip":
        a, b = _seconds(args.start), _seconds(args.end)
        if not 2 <= b - a <= 30:
            raise SystemExit("A voice sample should be 2-30 s of one person talking alone (5-10 s is ideal).")
        audio.clip(args.audio, args.out, a, b)
        print(f"Wrote {args.out} ({b - a:.0f} s). Use it with: --voice \"Name={args.out}\"")
        return 0

    if args.cmd == "shrink":
        dst = args.out or args.audio.with_suffix(".small.ogg")
        audio.shrink(args.audio, dst, args.bitrate)
        mb = lambda p: p.stat().st_size / 1e6  # noqa: E731
        print(f"{args.audio} ({mb(args.audio):.1f} MB) -> {dst} ({mb(dst):.1f} MB), "
              f"{audio.duration(dst) / 60:.1f} min")
        return 0

    if re.match(r"https?://", args.audio):
        from .fetch import download

        args.audio = download(args.audio, Path("downloads"))
    args.audio = Path(args.audio)
    if not args.audio.exists():
        raise SystemExit(f"No such file: {args.audio}")
    cfg.update(
        context=args.context, llm_model=args.llm_model, reasoning_effort=args.reasoning_effort,
        enhance=args.enhance, chunk_seconds=args.chunk_seconds, name_threshold=args.name_threshold,
        keywords=[k.strip() for k in args.keywords.split(",") if k.strip()] if args.keywords else None,
        voiceprints=False if args.no_voiceprints else None,
        min_speaker_seconds=args.min_speaker_seconds,
        trim_start=_seconds(args.start) if args.start else None,
        trim_end=_seconds(args.end) if args.end else None,
        rename=_renames(args.rename),
    )
    voices = _voices(args.voice)
    if voices:
        # Names of known participants are also the most useful ASR keywords.
        cfg.keywords = list(dict.fromkeys([*cfg.keywords, *voices]))
    out = args.out or Path("output") / args.audio.stem
    from .pipeline import run

    _interruptible(lambda: run(args.audio, out, cfg, voices=voices, fresh=args.fresh))
    print(f"\nWrote {out / 'transcript.md'} (+ .json, .srt, .txt)")
    return 0


# Run in a worker thread so Ctrl+C works on Windows even while a network call is blocking
def _interruptible(fn) -> None:
    import threading

    box: dict = {}

    def target():
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    t = threading.Thread(target=target, daemon=True)
    t.start()
    try:
        while t.is_alive():
            t.join(0.5)
    except KeyboardInterrupt:
        print("\nStopped. Finished steps are saved; run the same command again to resume.", flush=True)
        os._exit(130)  # skip waiting for in-flight API calls; cache files are written atomically
    if "error" in box:
        err = box["error"]
        text = str(err).lower()
        if "credit" in text or "insufficient_quota" in text or "billing" in text:
            print("\nYour OpenAI account is out of credits. Add credits at "
                  "https://platform.openai.com/settings/organization/billing and run the same command "
                  "again: finished steps are saved and won't be paid for twice.", flush=True)
            raise SystemExit(2)
        raise err


def check(cfg: Settings) -> int:
    ok = True
    print(f"ffmpeg: {audio.ffmpeg_exe()}")
    print(f"voiceprints extra: {'installed' if voiceprint.available() else 'not installed (optional)'}")
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY: missing (put it in .env or your environment)")
        return 1
    from .net import make_client

    client = make_client(max_retries=0)
    try:
        have = {m.id for m in client.models.list()}
    except Exception as exc:  # noqa: BLE001
        print(f"API: cannot list models: {exc}")
        return 1
    for m in (cfg.diarize_model, cfg.transcribe_model, cfg.llm_model):
        print(f"model {m}: {'ok' if m in have else 'NOT AVAILABLE to this key'}")
        ok &= m in have
    try:
        client.responses.create(model=cfg.llm_model, input="Reply with: ok", max_output_tokens=16)
        print("API billing: ok")
    except Exception as exc:  # noqa: BLE001
        print(f"API call failed: {exc}")
        ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
