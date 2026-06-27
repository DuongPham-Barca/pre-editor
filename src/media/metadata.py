import json
from dataclasses import dataclass

from src.media.ffmpeg import run_command


@dataclass
class VideoMetadata:
    width: int
    height: int
    fps: float
    duration: float
    video_codec: str | None
    audio_codec: str | None


def _parse_fps(value: str) -> float:
    if "/" in value:
        num, den = value.split("/")
        return float(num) / float(den)

    return float(value)


def get_video_metadata(video_path: str) -> VideoMetadata:
    command = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]

    output = run_command(command)
    data = json.loads(output)

    video_stream = None
    audio_stream = None

    for stream in data["streams"]:
        if stream.get("codec_type") == "video" and video_stream is None:
            video_stream = stream

        if stream.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = stream

    if video_stream is None:
        raise ValueError("Không tìm thấy video stream.")

    return VideoMetadata(
        width=int(video_stream["width"]),
        height=int(video_stream["height"]),
        fps=_parse_fps(video_stream.get("avg_frame_rate", "0/1")),
        duration=float(data["format"]["duration"]),
        video_codec=video_stream.get("codec_name"),
        audio_codec=audio_stream.get("codec_name") if audio_stream else None,
    )