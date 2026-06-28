import json
from pathlib import Path

from src.media.ffmpeg import run_command
from src.timeline.schemas import TimelineClip


def render_preview(
    input_video: str,
    timeline: list[TimelineClip],
    output_video: str,
    temp_dir: str = "temp/render",
) -> Path:
    output_path = Path(output_video)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not timeline:
        output_path.write_text("", encoding="utf-8")
        return output_path

    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)

    clip_files: list[Path] = []

    for index, clip in enumerate(timeline):
        clip_path = temp_path / f"clip_{index:04d}.mp4"

        command = [
            "ffmpeg",
            "-y",
            "-ss",
            str(clip.source_start),
            "-to",
            str(clip.source_end),
            "-i",
            input_video,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-preset",
            "veryfast",
            str(clip_path),
        ]

        run_command(command)
        clip_files.append(clip_path)

    concat_file = temp_path / "concat.txt"

    concat_file.write_text(
        "\n".join(
            f"file '{clip_file.resolve().as_posix()}'"
            for clip_file in clip_files
        ),
        encoding="utf-8",
    )

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(output_path),
    ]

    run_command(command)

    return output_path