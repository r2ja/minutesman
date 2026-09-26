# State of the art, cost, and model choice (September 2026)

Goal: an 80-minute phone recording of meetings in several rooms (clean, far-field,
quiet and distorted stretches), with Urdu and English mixed mid-sentence. Output: a diarized
transcript in Roman Urdu + English, with a confidence per speaker attribution, real names
where the audio supports them and `Guest N` otherwise. OpenAI platform.

Prices are list prices from <https://developers.openai.com/api/docs/pricing>, checked
2026-09-26. Model availability was checked against this account's `/v1/models`.

## 1. What OpenAI offers today

| Model | What it does | Price | Relevance |
|---|---|---|---|
| `gpt-transcribe` (Jul 2026) | Best OpenAI file ASR. `languages` hints (several allowed), `keywords`, `prompt`, handles code-switching, reports detected languages. **No diarization, no timestamps.** | $0.0045/min | Best text quality → **pass B** |
| `gpt-4o-transcribe-diarize` | ASR + speaker labels + segment timestamps (`diarized_json`). Takes up to **4 known-speaker reference clips** (2-10 s). No `prompt`. | $0.006/min | Only OpenAI model that says *who spoke when* → **pass A** |
| `gpt-4o-transcribe` / `-mini` | Previous-generation ASR | $0.006 / $0.003 | Fallback if `gpt-transcribe` is unavailable |
| `whisper-1` | Original Whisper; the only one with word timestamps | $0.006/min | Not needed; roughly double `gpt-transcribe`'s error rate on multilingual audio |
| `gpt-live-transcribe`, `gpt-realtime-whisper` | Streaming | $0.017/min | Real-time only; ~4× the price for no gain on files |
| `gpt-audio-1.5` | LLM that listens to audio | $32 / 1M audio-in tokens | Could "listen" to hard segments, but ~10× the cost of the whole pipeline; not used by default |
| `gpt-6-astra` | Top reasoning LLM | $10 in / $50 out per 1M | Overkill for transcript cleanup |
| **`gpt-6-sol`** | Mid-tier GPT-6 with reasoning, structured outputs | $2 in / $10 out | **Fusion + Roman Urdu + naming** |
| `gpt-6-luna` | Cheapest GPT-6 | $0.10 / $0.50 | Budget option; weaker at romanizing ambiguous Urdu |

Reported quality: OpenAI's figures cite `gpt-transcribe` at ~19% WER against whisper-1's ~40% on
Common Voice across 22 languages (reported by [Spokenly's write-up](https://spokenly.app/blog/gpt-transcribe)).
Urdu-specific numbers are not published. **Nothing here has been measured on your audio yet.**

## 2. Why two passes plus an LLM

No single OpenAI model does all of it:

- The diarizer is the only source of *who* and *when*, but its text is older-generation, it takes
  no prompt/keywords, and in Urdu it tends to switch script (Urdu Nastaliq, sometimes Hindi
  Devanagari for the same words).
- `gpt-transcribe` produces better code-switched text but returns one text blob with no
  timestamps. Running it on **~90-second windows cut at the diarizer's turn boundaries** keeps
  it aligned, so the LLM only has to match one window's text against that window's segments.
- ASR models transcribe Urdu in Urdu script. **Roman Urdu is a transliteration job**, which needs
  an LLM anyway. The same call reconciles the two hypotheses (dropping hallucinations common
  on quiet/far audio, filling words one pass dropped) and checks each speaker label from the
  conversation itself (a question answered by the "same" speaker is suspicious).

## 3. Speaker identity across an 80-minute file

The diarizer labels speakers per request (`A`, `B`, ...), and an 80-minute file must be chunked
(25 MB upload cap; chunks are ~10 min, sent as 64 kbps mono MP3, ~4.8 MB). The labels are
stitched together with, in order of strength:

1. **Known-speaker references.** After each chunk the best 3-10 s clip of every speaker is kept.
   The next chunk is sent with up to 4 of them (enrolled > most recent > most talkative), and
   the diarizer answers with our global ids.
2. **Overlap voting.** Chunks overlap by 30 s; a new label talking over the same seconds as a
   known speaker in the previous chunk is that speaker.
3. **Voiceprints (optional, local, free).** SpeechBrain ECAPA-TDNN embeddings per segment:
   merge ids that are clearly one voice, give every segment an **acoustic confidence**, and
   flag or move segments whose voice fits another speaker much better. CPU only, ~1 GB install.
   More than 4 recurring speakers is exactly where signal 1 runs out and this one matters.

**Names:** a final LLM pass reads the whole transcript and names a speaker only on explicit
evidence (self-introduction, being addressed right before replying). Below 0.75 confidence
the speaker stays `Guest N` and the guess is shown in the speaker table ("maybe Sara (0.55)").
Voice samples passed with `--voice Name=file.wav` are exact: they go to the diarizer as
references and to the voiceprint matcher.

**Confidence per segment** = weighted blend, renormalized over whatever is available:
voiceprint 0.5, chunk-link 0.3, LLM conversation check 0.2. Then ×0.8 if under 1 s and ×0.85
if the original audio there is quieter than -40 dBFS. Turns under 0.6 are marked ⚠.

## 4. Audio conditions

`--enhance light` (default) runs in ffmpeg: band-limit → **dynamic gain first** (up to 30×, so far and
quiet talkers are lifted) → mild spectral denoise → final level trim. Order matters:
a denoiser with a fixed noise floor erases speech sitting at -45 dBFS if it runs before the gain
(caught by this repo's tests). Heavy denoising is avoided because it creates artifacts that hurt ASR
more than the noise does. `strong` adds a compressor for very uneven recordings; `off` sends the audio as recorded.

## 5. Cost for one 80-minute recording

`minutesman estimate 80` (assumes ~150 words/min):

| LLM (effort) | Pass A diarize | Pass B text | LLM | **Total** |
|---|---|---|---|---|
| `gpt-6-luna` (low) | $0.50 | $0.36 | $0.03 | **≈ $0.90** |
| **`gpt-6-sol` (medium), default** | $0.50 | $0.36 | $0.74 | **≈ $1.61** |
| `gpt-6-astra` (medium) | $0.50 | $0.36 | $3.72 | **≈ $4.58** |

Voiceprints cost nothing (local CPU). Reruns reuse cached API results, so after the first run,
changing names or thresholds costs only the LLM calls that changed. A real run writes its actual
token counts and cost to `transcript.json → meta.cost`.

For comparison, `gpt-audio-1.5` "listening" to all 80 minutes would be roughly $1.50-3 in audio
tokens alone before any output. That is why it isn't in the default path.

## 6. Alternatives considered (outside OpenAI)

- **WhisperX / faster-whisper + pyannote locally**: free per run, but needs a GPU for sensible
  speed on 80 min. Whisper large-v3 Urdu output is in Urdu script and weaker on code-switching,
  and pyannote still needs cross-chunk handling on long far-field audio. Its embeddings idea is
  what the optional voiceprint stage borrows.
- **Other hosted ASR with diarization** (ElevenLabs Scribe, AssemblyAI, Deepgram, Google Chirp,
  Gemini audio): viable, not priced or tested here because the brief is OpenAI. The pipeline's
  pass A / pass B split makes swapping a pass straightforward.

## 7. Decision

- Pass A: `gpt-4o-transcribe-diarize`, 10-min chunks, 30 s overlap, rolling known-speaker refs.
- Pass B: `gpt-transcribe` with `languages=[ur, en]`, meeting context, keywords, on ~90 s windows.
- Fusion + Roman Urdu + speaker naming: `gpt-6-sol`, reasoning effort medium, structured outputs.
- Voiceprints: on when installed (recommended on the PC; optional in the cloud).
- Everything is configurable (`--llm`, `--effort`, env vars) to trade cost against quality after
  seeing real results.
