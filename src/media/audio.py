from pathlib import Path

from src.media.ffmpeg import run_command


def extract_audio(
    input_video: str,
    output_audio: str,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Path:
    output_path = Path(output_audio)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_video,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(output_path),
    ]

    run_command(command)

    return output_path