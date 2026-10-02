# minutesman: context for Claude

Personal tool of Raja Ata Ul Karim (GitHub r2ja). It turns long phone recordings of office meetings
in Urdu/English (code-switched) into diarized transcripts in Roman Urdu + English, with a speaker
confidence per turn, real names only when the audio supports them (else "Guest N"), and one section
per meeting. Built and tested in a cloud session on 2026-09-26; now runs locally on Windows.

## House rules (from the owner)

- Commit as the owner, never as Claude: `Raja Ata Ul Karim <69019157+r2ja@users.noreply.github.com>`.
  No Co-Authored-By or other Claude attribution lines.
- Commit messages: one line.
- Code comments: simple single-line `#` comments. No docstrings, no multi-line comment blocks.
- Work branch: `claude/serene-curie-kk6ek4`. `main` does not exist yet; the owner may squash-merge.
  The first three commits on the branch are authored by Claude (a history rewrite was not done).

## How it runs

- Windows: `setup.bat` (venv + install + API key into `.env` + `minutesman check`), then drag a
  recording onto `transcribe.bat` (cost estimate, run, opens the transcript).
- CLI: `minutesman run <file or share link> [--keywords "names,terms"] [--voice Name=clip.wav]
  [--context "..."] [--enhance off|light|strong|rnnoise|deepfilter] [--llm gpt-6-sol] [--effort medium]`,
  `minutesman estimate <file|minutes>`, `minutesman check`, `minutesman shrink <file>`.
- Output: `output/<file stem>/transcript.md|.txt|.srt|.json`. `work/` holds the enhanced audio and
  cached API results, so reruns resume and never pay twice (`--fresh` redoes everything).
- Tests: `pytest` (27, offline, fake OpenAI client that recognises tone "speakers" by pitch).
  Lint: `ruff check --line-length 120 minutesman tests scripts`.

## Pipeline (minutesman/)

1. `audio.py`: bundled ffmpeg (imageio-ffmpeg). `light` enhance = band-limit, dynaudnorm gain FIRST
   (lifts far/quiet talkers), mild afftdn, trim. Denoising before gain erased -45 dBFS speech (bug found).
2. `asr.py` pass A: `gpt-4o-transcribe-diarize`, 10-min chunks, 30 s overlap, sequential, each chunk gets
   up to 4 known-speaker reference clips (API limit) so labels stay consistent.
3. `speakers.py`: links chunk labels to global ids (references, then overlap voting).
4. `voiceprint.py` (optional extra, SpeechBrain ECAPA, CPU): re-clusters segments >= 1.5 s by voice
   (average linkage, cut 0.30, +0.10 prior for existing links), maps clusters back to ids, short segments
   follow their nearest same-label neighbour. Fixes the diarizer merging different people in the same room.
5. pass B: `gpt-transcribe` (languages ur,en + keywords + prompt) on ~90 s windows cut at turn boundaries.
6. `llm.py` fusion: `gpt-6-sol` (Responses API, structured outputs, reasoning medium) merges A+B per chunk
   into Roman Urdu/English and sanity-checks speaker labels from the conversation.
7. `llm.analyze`: one call over the whole transcript: meeting boundaries + names judged PER MEETING.
   `pipeline.resolve_names` splits an id confidently named differently in two meetings.
8. Confidence = voiceprint 0.5 + linking 0.3 + LLM 0.2 (renormalised), x0.8 if < 1 s, x0.85 if < -40 dBFS.
   Names published only at >= 0.75. `render.py` keeps low-confidence (< 0.6) segments as separate ⚠ turns.

## What was measured (synthetic TTS meetings, scripts/make_test_audio.py + scripts/evaluate.py)

- Text: ~3-3.5% character error vs a hand-written Roman Urdu reference, whatever the enhancement.
- Neural denoising (RNNoise, DeepFilterNet) did not help speaker attribution; Krisp is enterprise-only.
  Default stays `light`; worth comparing `--enhance off` on real audio.
- Single meeting, 4 speakers: 62-77% of speech time attributed correctly; errors cluster on two similar
  female voices in the same degraded room, and those turns mostly get low confidence.
- Two meetings + hallway walk: 2/2 meetings found, 10/10 turns correct after per-meeting naming,
  3.1% CER, $0.054.
- Cost: ~$0.026 per audio minute with gpt-6-sol (estimator in cli.py is calibrated on real runs).
- Nothing has been validated on real recordings yet; thresholds were tuned on synthetic audio only.

## Current state

- First real run ("Systems limited.m4a", 109.4 min, 12 chunks) completed 2026-09-26 21:02 PKT after the
  streaming fix and a credit top-up. Pass A ~4.5 min per chunk once streamed; voiceprints 1006 segments in
  ~11 min CPU (cached now), 25 voice groups, 314 segments moved, 14 extra speakers; analysis over 1746
  segments took ~30 s. Output in output/Systems limited.m4a/.
- Review of that run: 2 meetings found (plausible), no Urdu/Devanagari script leaked, only 2.3 of 109 min
  flagged below 0.6. Problems: 28 speakers where ~8 are real (20 "Guests" had < 5 s each: backchannels and
  sentence tails split off by voiceprints) -> `absorb_minor_speakers` folds speakers under
  `min_speaker_seconds` (15) into the nearest speaker (28 -> 9 on that data). Meeting titles came out in
  Urdu script -> prompt now asks for English/Roman Urdu. The owner is the top talker but unnamed ->
  `--rename "S2=Name"` sets names by speaker id with no API cost. Speaker ids were `len(entries)+1`, which
  could reuse a dropped id and merge two people -> monotonic counters; voice samples use E ids.
- Off-meeting stretches (side phone/Teams calls on the recording phone, stepping out, tail after the meeting):
  `llm.analyze` returns `off_meeting` spans with a reason; `pipeline.mark_off_meeting` tags segments
  (`off_reason`). transcript.md/.txt hide them behind one-line markers, transcript_full.md keeps all.
  `--start/--end` trim the audio (own cache dir per start trim; timestamps shifted back to the original).
  transcribe.bat asks for context, names/terms, end/start time and saves answers in run_options.txt.
- Fragment folding uses min(15 s, 2% of all speech) so short recordings keep short real speakers;
  meeting participants are recounted after name resolution (it can split ids).
- Client recordings are confidential: never commit transcripts or quote their content in the repo.
- Voiceprint embeddings are cached in work/cache/voiceprints.npz; per-segment moves log at DEBUG only.

## Next steps / open ideas

- Review the first real transcript with the owner: speaker mix-ups, names, meeting split, Roman Urdu
  spelling. Tune voiceprint thresholds (`LINK_THRESHOLD`, `PRIOR_BONUS`) and `name_threshold` on real audio.
- Speed: pass A is the bottleneck (sequential because of rolling speaker references). Option: run chunks in
  parallel without references and rely on overlap voting + voiceprints; measure accuracy before switching.
- Progress: heartbeat every 30 s while waiting on API calls ("heard up to m:ss of 10:30" in pass A),
  per-chunk time + ETA, counters for pass B / fusion / voiceprints (`progress.py`).
- Ctrl+C: `cli._interruptible` runs the pipeline in a worker thread so Ctrl+C works on Windows mid-request.
- Hang root cause (first real run): every non-streamed request that finished got back in under ~4 min;
  every one that would take longer never returned (chunk 3 once, chunk 5 twice). A ~4 min idle-connection
  cutoff on the path (Azure LB default / home router) drops silent connections. Fix: pass A and all LLM
  calls now STREAM (diarize emits transcript.text.segment events every few seconds), all connections use
  TCP keepalive (`net.make_client`), and timeouts are "no data for 240 s" instead of a total time.
  Hedging (`progress.hedged`) stays as a backstop. Cache entries record their chunk span.
- Known leftovers: very short greetings can become their own flagged "Guest"; the person recording is
  often never named (nobody addresses them right before they speak); `--voice` clips fix both.
