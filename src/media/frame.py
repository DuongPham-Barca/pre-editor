from pathlib import Path

from src.media.ffmpeg import run_command


def extract_frame(
    input_video: str,
    timestamp: float,
    output_image: str,
) -> Path:
    output_path = Path(output_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(timestamp),
        "-i",
        input_video,
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]

    run_command(command)

    return output_path