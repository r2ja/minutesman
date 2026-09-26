# minutesman

Diarized meeting transcripts for **Urdu + English (code-switched)** recordings, written in
**Roman Urdu + English**. Every turn carries a **speaker confidence**; speakers get real names
only when the audio or conversation supports it, otherwise `Guest 1`, `Guest 2`, ...

Built for messy phone recordings: far-away, quiet, distorted, and changing rooms mid-file.
It runs the same way on a PC (Windows/macOS/Linux) and in a cloud container, needs only
Python, and uses the bundled ffmpeg from `imageio-ffmpeg`.

Output format (illustrative):

```
**[00:14:03] Ahmed** (0.91): Okay, lekin kya is se lead generation par asar parega?
**[00:14:08] Guest 2** (0.84): Mera khayal hai nahi. Hum organic channels par zyada focus kar rahe hain, and the numbers look stable so far.
**[00:14:17] Sara** (0.58) ⚠: Bilal, can you share the dashboard link after the meeting?
```

Why these models, what it costs (about **$2 per 80-minute recording** by default, measured), and how
speaker identity and confidence work: see **[docs/ANALYSIS.md](docs/ANALYSIS.md)**.

## Pipeline

```
audio ─► ffmpeg enhance (gain first, mild denoise) ─► 10-min chunks, 30 s overlap
   ├─► Pass A  gpt-4o-transcribe-diarize  who/when (+ rolling known-speaker reference clips)
   ├─► speaker linking across chunks (references → overlap vote → voiceprints*)
   ├─► Pass B  gpt-transcribe on ~90 s windows  better code-switched text (ur+en hints)
   ├─► LLM fusion  gpt-6-sol  A+B → Roman Urdu/English, per-segment speaker sanity check
   └─► LLM analysis  meeting boundaries + names per meeting (≥ 0.75 else "Guest N")
        → transcript.md / .txt / .srt / .json, split by meeting
* optional local ECAPA speaker embeddings (free, CPU)
```

## Setup

Python 3.10+.

```bash
git clone https://github.com/r2ja/minutesman.git
cd minutesman
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -e .
# Recommended on your PC: local voiceprints for acoustic confidence (~1 GB, CPU is fine)
pip install -e ".[voiceprint]"

cp .env.example .env        # Windows: copy .env.example .env
# edit .env and set OPENAI_API_KEY
minutesman check            # ffmpeg, key, model access, billing
```

In a cloud container, set `OPENAI_API_KEY` as an environment secret instead of `.env`.

## Use

```bash
minutesman estimate path/to/recording.m4a        # cost before spending anything
minutesman run path/to/recording.m4a \
    --context "Office day in Lahore, several meetings" \
    --keywords "Ahmed,Sara,Bilal,Ayesha,Jira,staging,Q3"
```

A file in your Downloads folder:

```bash
minutesman run "%USERPROFILE%\Downloads\recording.m4a"     # Windows (cmd)
minutesman run "$HOME\Downloads\recording.m4a"             # Windows (PowerShell)
minutesman run ~/Downloads/recording.m4a                   # macOS / Linux
```

Any size and format that ffmpeg reads works (m4a, mp3, wav, aac, ogg, even video). Chunks
are re-encoded before upload, so a 120 MB file never hits the API's 25 MB limit.

### One recording, several meetings

The analysis step finds where each meeting starts and ends: greetings, closings, long silences,
a change in who is present or in the topic. The transcript gets a **Meetings** table and one
section per meeting. Walking and hallway talk go under **Between meetings**, and long silences
are marked. Names are judged per meeting. If the diarizer gave two similar voices from
different rooms the same id and the conversation names them differently (e.g. "Sara" in one
room, "Ayesha" in the next), the id is split into two people.

### Running in the cloud

A recording is too big (and too private) for git. Make a small speech-quality copy, share it
as "anyone with the link" on Google Drive or Dropbox, and pass the link:

```bash
minutesman shrink recording.m4a            # -> recording.small.ogg, ~19 MB for 80 min
minutesman run "https://drive.google.com/file/d/<id>/view?usp=sharing"
```

Output goes to `output/<file name>/`:

| File | Contents |
|---|---|
| `transcript.md` | Speaker table (with name evidence), meetings table, and the transcript per meeting with confidences; ⚠ on turns below 0.60 |
| `transcript.txt` | Same, plain text |
| `transcript.srt` | Subtitles with speaker labels |
| `transcript.json` | Everything: segments with both ASR hypotheses, all confidence components, notes, cost and token usage |
| `work/` | Enhanced audio and cached API results (reruns are free for finished stages) |

### Getting names right

- **Voice samples** (best): 5-10 s of a person speaking alone, cut from any recording.
  `--voice "Ahmed=voices/ahmed.wav" --voice "Sara=voices/sara.m4a"`. Up to 4 are sent to the
  diarizer as references, and any number are matched by voiceprints.
- **Keywords**: participant names and jargon improve the text *and* the name inference.
- Otherwise names come from the conversation ("Sara, aap batayein?" followed by a reply). Weak
  evidence stays `Guest N`, and the guess is shown in the speaker table.

### Useful options

| Flag / env var | Default | Notes |
|---|---|---|
| `--llm` / `MINUTESMAN_LLM_MODEL` | `gpt-6-sol` | ≈ $2.06 per 80 min in total; `gpt-6-luna` ≈ $0.93, `gpt-6-astra` ≈ $6.84 |
| `--effort` / `MINUTESMAN_REASONING_EFFORT` | `medium` | `low` is cheaper, `high` is more careful |
| `--enhance` / `MINUTESMAN_ENHANCE` | `light` | `off`, `strong`, or neural denoisers `rnnoise` / `deepfilter` (DeepFilterNet, downloaded on first use). Measured: denoising didn't improve results on the test meeting, see [ANALYSIS §4](docs/ANALYSIS.md#4-audio-conditions-and-noise-cancellation) |
| `--name-threshold` | `0.75` | Minimum confidence to publish a real name |
| `--no-voiceprints` | | Skip local embeddings even if installed |
| `--chunk-seconds` | `600` | Diarization chunk length |
| `--fresh` | | Ignore cached API results |
| `MINUTESMAN_LANGUAGES` | `ur,en` | Language hints for pass B |

### Test audio

`python scripts/make_test_audio.py` synthesizes a ~1.5-minute, 4-speaker Urdu/English meeting
(`--scenario multi`: two meetings, 5 people, a hallway walk and a long silence)
with OpenAI TTS (costs about a cent), degrades parts of it (far, low, crispy), and writes a
ground-truth file next to it:

```bash
python scripts/make_test_audio.py --out samples/synthetic_meeting.wav
minutesman run samples/synthetic_meeting.wav -o output/synth --keywords "Ahmed,Sara,Bilal,Ayesha"
python scripts/evaluate.py output/synth/transcript.json samples/synthetic_meeting.truth.json
```

`evaluate.py` reports speaker accuracy (overall and per recording condition), the number of
speakers found, and the character error rate against the Roman Urdu reference.

## Development

```bash
pip install -e ".[dev]"
pytest                      # offline: fake OpenAI client, real ffmpeg
```

The tests use a fake client that recognizes synthetic "speakers" by pitch, including in the
reference clips. That exercises chunking, overlap de-duplication, cross-chunk linking, caching,
enrollment, labels and output without spending API credit.
