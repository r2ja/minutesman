"""Generate a synthetic Urdu/English code-switched meeting for testing.

Uses OpenAI TTS with a distinct voice per speaker, then degrades sections to
mimic a phone recording moving between rooms (far-field, low gain, crackle).

    python scripts/make_test_audio.py --out samples/synthetic_meeting.wav
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI

from minutesman.audio import duration, ffmpeg_exe  # needs `pip install -e .`

# (speaker, voice, condition, text for TTS, Roman Urdu reference). Urdu is given to the
# TTS in Urdu script so it is pronounced naturally; the reference is what a perfect
# transcript would say. Mid-sentence language switches are included on purpose.
SCRIPT = [
    ("Ahmed", "onyx", "clean",
     "Assalam o alaikum everyone. چلیں شروع کرتے ہیں۔ Today's agenda mein teen cheezein hain: Q3 budget, hiring plan, aur client demo.",
     "Assalam o alaikum everyone. Chalein shuru karte hain. Today's agenda mein teen cheezein hain: Q3 budget, hiring plan, aur client demo."),
    ("Sara", "nova", "clean",
     "Thanks Ahmed. Budget ke hawale se, ہم نے marketing spend تقریباً twenty percent کم کر دیا ہے۔",
     "Thanks Ahmed. Budget ke hawale se, hum ne marketing spend taqreeban twenty percent kam kar diya hai."),
    ("Ahmed", "onyx", "clean",
     "Okay, لیکن کیا اس سے lead generation پر اثر پڑے گا؟",
     "Okay, lekin kya is se lead generation par asar parega?"),
    ("Bilal", "echo", "far",
     "Mera khayal hai nahi. ہم organic channels پر زیادہ focus کر رہے ہیں، and the numbers look stable so far.",
     "Mera khayal hai nahi. Hum organic channels par zyada focus kar rahe hain, and the numbers look stable so far."),
    ("Sara", "nova", "far",
     "Bilal, can you share the dashboard link after the meeting? مجھے weekly trend دیکھنا ہے۔",
     "Bilal, can you share the dashboard link after the meeting? Mujhe weekly trend dekhna hai."),
    ("Bilal", "echo", "far",
     "Haan bilkul, I'll send it on Slack.",
     "Haan bilkul, I'll send it on Slack."),
    ("Ayesha", "shimmer", "low",
     "Hiring ke baare mein, ہمیں دو backend engineers چاہئیں، ideally by end of October.",
     "Hiring ke baare mein, hamein do backend engineers chahiyein, ideally by end of October."),
    ("Ahmed", "onyx", "low",
     "Ayesha, budget approved hai? کیونکہ finance نے ابھی sign off نہیں کیا۔",
     "Ayesha, budget approved hai? Kyunke finance ne abhi sign off nahi kiya."),
    ("Ayesha", "shimmer", "crispy",
     "Nahi abhi pending hai. I'll follow up with finance tomorrow, انشاءاللہ۔",
     "Nahi abhi pending hai. I'll follow up with finance tomorrow, InshaAllah."),
    ("Sara", "nova", "crispy",
     "Aur client demo Thursday ko hai, right? ہمیں staging environment کل تک ready چاہیے۔",
     "Aur client demo Thursday ko hai, right? Hamein staging environment kal tak ready chahiye."),
    ("Bilal", "echo", "crispy",
     "Staging is ready. بس ایک bug باقی ہے login page پر، I'll fix it tonight.",
     "Staging is ready. Bas aik bug baqi hai login page par, I'll fix it tonight."),
    ("Ahmed", "onyx", "clean",
     "Perfect. تو action items: Bilal dashboard share karega, Ayesha finance se follow up, aur demo Thursday. Thank you sab ka.",
     "Perfect. To action items: Bilal dashboard share karega, Ayesha finance se follow up, aur demo Thursday. Thank you sab ka."),
]

INSTRUCTIONS = (
    "Speak naturally like a Pakistani professional in an office meeting, "
    "switching between Urdu and English with a Pakistani accent."
)

# ffmpeg filter chains that simulate the recording conditions.
CONDITIONS = {
    "clean": "anull",
    "far": "lowpass=f=3000,aecho=0.8:0.7:60|120:0.35|0.25,volume=0.35",
    "low": "volume=0.12,highpass=f=200",
    "crispy": "highpass=f=400,lowpass=f=3400,acrusher=bits=8:mix=0.35,volume=1.6",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples/synthetic_meeting.wav")
    ap.add_argument("--model", default="gpt-4o-mini-tts")
    args = ap.parse_args()

    client = OpenAI()
    ff = ffmpeg_exe()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    truth = []

    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        t = 0.0
        for i, (speaker, voice, cond, text, roman) in enumerate(SCRIPT):
            raw = Path(tmp) / f"{i:02d}_raw.wav"
            with client.audio.speech.with_streaming_response.create(
                model=args.model, voice=voice, input=text,
                instructions=INSTRUCTIONS, response_format="wav",
            ) as resp:
                resp.stream_to_file(raw)
            proc = Path(tmp) / f"{i:02d}.wav"
            noise = "anoisesrc=color=pink:amplitude=0.02" if cond != "clean" else "anoisesrc=amplitude=0.001"
            subprocess.run(
                [ff, "-y", "-loglevel", "error", "-i", str(raw), "-f", "lavfi", "-i", noise,
                 "-filter_complex",
                 f"[0:a]aresample=16000,{CONDITIONS[cond]}[s];[1:a]aresample=16000[n];"
                 f"[s][n]amix=inputs=2:duration=first:normalize=0,apad=pad_dur=0.6",
                 "-ac", "1", "-ar", "16000", str(proc)],
                check=True,
            )
            parts.append(proc)
            total = duration(proc)
            speech = total - 0.6  # apad adds 0.6 s of silence after each turn
            truth.append({"speaker": speaker, "condition": cond, "start": round(t, 2),
                          "end": round(t + speech, 2), "text": text, "reference": roman})
            t += total
            print(f"[{i + 1}/{len(SCRIPT)}] {speaker} ({cond})")

        listfile = Path(tmp) / "list.txt"
        listfile.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
        subprocess.run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", str(listfile), "-c", "copy", str(out)], check=True)

    out.with_suffix(".truth.json").write_text(json.dumps(truth, ensure_ascii=False, indent=2))
    print(f"Wrote {out} and {out.with_suffix('.truth.json')}")


if __name__ == "__main__":
    main()
