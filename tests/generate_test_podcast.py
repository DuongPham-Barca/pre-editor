"""Generate a synthetic test podcast with known script and audio.

Creates:
  - output/test/podcast.mp4       video with 2-channel audio
  - output/test/podcast_script.txt full script with timestamps
  - output/test/podcast_plan.txt   podcast plan for RAG filter
  - output/test/track_a.wav        isolated speaker A audio
  - output/test/track_b.wav        isolated speaker B audio

Run:  python tests/generate_test_podcast.py
"""

import struct
import wave
from pathlib import Path

import numpy as np

SR = 16000
OUT = Path("output/test")
OUT.mkdir(parents=True, exist_ok=True)


def _sin(freq: float, duration: float, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return 0.4 * np.sin(2 * np.pi * freq * t)


def _noise(duration: float, sr: int = SR) -> np.ndarray:
    return np.random.uniform(-0.05, 0.05, int(sr * duration))


def _silence(duration: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(sr * duration))


def _save_wav(path: Path, data: np.ndarray, sr: int = SR):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((data * 32767).clip(-32767, 32767).astype(np.int16).tobytes())


def _save_mp4(video_path: Path, track_a: np.ndarray, track_b: np.ndarray, sr: int = SR):
    """Create a minimal MP4 with black video and stereo audio (A=left, B=right)."""
    import subprocess

    duration = max(len(track_a), len(track_b)) / sr
    stereo = np.column_stack([
        track_a / np.max(np.abs(track_a)) if np.max(np.abs(track_a)) > 0 else track_a,
        track_b / np.max(np.abs(track_b)) if np.max(np.abs(track_b)) > 0 else track_b,
    ])
    wav_temp = OUT / "_temp_stereo.wav"
    _save_wav(wav_temp, stereo.flatten(), sr)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s=640x480:d={duration:.2f}:r=10",
        "-i", str(wav_temp),
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac", "-shortest",
        str(video_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    wav_temp.unlink()


# ══════════════════════════════════════════════
# Script
# ══════════════════════════════════════════════

script: list[tuple[float, float, str, str]] = [
    # (start_sec, end_sec, speaker, text)

    # ── Intro (on-topic) ──
    (0, 8, "A", "Welcome to English Speaking Practice. Today we will focus on common workplace conversations."),
    (8, 15, "B", "That is right. We will practice ordering coffee, asking for directions, and small talk with colleagues."),
    (15, 20, "A", "These are essential skills for anyone working in an international environment."),

    # ── Dead air (silence) ──
    (20, 23, "", ""),

    # ── Core lesson (on-topic) ──
    (23, 30, "A", "Let us start with ordering coffee. The first phrase you need is: I would like a latte please."),
    (30, 38, "B", "Great. Then the barista might ask: Would you like that hot or iced? You can say: Hot please."),
    (38, 43, "A", "Now let us practice together. Repeat after me: I would like a cappuccino."),
    (43, 48, "B", "Excellent pronunciation. Next we will cover asking for directions."),

    # ── Cross-talk (both speak) ──
    (48, 52, "AB", "So the train station is Excuse me where is the train station?"),
    (52, 58, "A", "The phrase is: Excuse me, where is the train station? Very useful when traveling."),

    # ── Off-topic tangent ──
    (58, 68, "B", "You know, last week I saw a movie about time travel. It was fascinating. The director really captured the paradoxes well."),
    (68, 75, "A", "Oh I love that film. The special effects were incredible. Anyway, back to our English lesson."),

    # ── Core lesson continues ──
    (75, 83, "A", "For asking directions you also need to understand the reply: Go straight and turn left at the traffic lights."),
    (83, 90, "B", "Exactly. Then the person might say: It is on your right, next to the bank. That is very helpful."),
    (90, 95, "AB", "Perfect. Let us practice that exchange together."),

    # ── Dead air ──
    (95, 97, "", ""),

    # ── Small talk section ──
    (97, 105, "B", "Now for small talk. A common question is: How was your weekend? You can reply: It was great, I went hiking."),
    (105, 112, "A", "Another good question is: Have you tried the new restaurant downtown? That is a great conversation starter."),

    # ── Off-topic: generic thanks ──
    (112, 118, "A", "Thank you so much for watching. Please like and subscribe. Also special thanks to our sponsors."),

    # ── Wrap-up (on-topic) ──
    (118, 125, "B", "We hope this lesson helps you feel more confident in workplace English conversations."),
    (125, 130, "A", "Keep practicing every day and you will improve quickly. See you next time."),
]

# ══════════════════════════════════════════════
# Generate audio
# ══════════════════════════════════════════════

total_duration = max(end for _, end, _, _ in script)
track_a = np.zeros(int(SR * total_duration))
track_b = np.zeros(int(SR * total_duration))
script_lines: list[str] = []

for start, end, speaker, text in script:
    dur = end - start
    if speaker == "A":
        track_a[int(SR * start):int(SR * end)] = _sin(220, dur) + _noise(dur)
    elif speaker == "B":
        track_b[int(SR * start):int(SR * end)] = _sin(340, dur) + _noise(dur)
    elif speaker == "AB":
        seg_a = _sin(220, dur) + _noise(dur)
        seg_b = _sin(340, dur) + _noise(dur)
        # Cross-talk: make them quieter so both are active
        track_a[int(SR * start):int(SR * end)] = seg_a * 0.6
        track_b[int(SR * start):int(SR * end)] = seg_b * 0.6
    # else silence — do nothing

    script_lines.append(f"[{start:.0f}s-{end:.0f}s] {speaker}: {text}")

# ── Save audio tracks ──
_save_wav(OUT / "track_a.wav", track_a)
_save_wav(OUT / "track_b.wav", track_b)
print(f"  Audio tracks: {OUT / 'track_a.wav'}, {OUT / 'track_b.wav'}")

# ── Save script ──
(OUT / "podcast_script.txt").write_text(
    "=== FULL SCRIPT ===\n\n" + "\n".join(script_lines),
    encoding="utf-8",
)
print(f"  Script:       {OUT / 'podcast_script.txt'}")

# ── Save podcast plan ──
(OUT / "podcast_plan.txt").write_text(
    "Keep sections that teach English conversation skills: "
    "ordering coffee, asking for directions, small talk.\n"
    "Exclude movie discussions, sponsor thanks, generic greetings.",
    encoding="utf-8",
)
print(f"  Podcast plan:  {OUT / 'podcast_plan.txt'}")

# ── Create video ──
try:
    _save_mp4(OUT / "podcast.mp4", track_a, track_b)
    print(f"  Video:        {OUT / 'podcast.mp4'}")
except Exception as e:
    print(f"  [WARN] Video creation failed: {e}")

print("\nDone! Now run:\n  python src/main.py")
