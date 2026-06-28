from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.media.audio import extract_audio
from src.media.ffmpeg import check_ffmpeg, run_command


VIDEO_EXTENSIONS = frozenset({".mp4", ".mov"})
AUDIO_EXTENSIONS = frozenset({".wav"})
SUPPORTED_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS
SMPTE_TIMECODE_PATTERN = re.compile(r"^\d{2,}:\d{2}:\d{2}[:;]\d{2}$")


@dataclass(frozen=True)
class TimecodeCandidate:
    value: str
    source: str


@dataclass(frozen=True)
class TimecodeMetadata:
    value: str | None
    source: str | None
    drop_frame: bool | None
    candidates: list[TimecodeCandidate]
    bwf_time_reference_samples: int | None


@dataclass(frozen=True)
class VideoStreamMetadata:
    index: int
    codec_name: str | None
    width: int | None
    height: int | None
    frame_rate: str | None
    average_frame_rate: str | None
    time_base: str | None
    start_time_seconds: float | None


@dataclass(frozen=True)
class AudioStreamMetadata:
    index: int
    codec_name: str | None
    sample_rate: int | None
    channels: int | None
    channel_layout: str | None
    time_base: str | None
    start_time_seconds: float | None


@dataclass(frozen=True)
class MediaProbe:
    format_name: str | None
    duration_seconds: float | None
    start_time_seconds: float | None
    video: VideoStreamMetadata | None
    audio: AudioStreamMetadata | None
    timecode: TimecodeMetadata
    format_tags: dict[str, str]


@dataclass(frozen=True)
class IngestedAsset:
    asset_id: str
    media_kind: str
    source_path: str
    source_relative_path: str
    source_extension: str
    file_size_bytes: int
    normalized_audio_path: str
    probe: MediaProbe


@dataclass(frozen=True)
class IngestManifest:
    schema_version: str
    created_at_utc: str
    input_directory: str
    output_directory: str
    audio_spec: dict[str, Any]
    assets: list[IngestedAsset]


def scan_input_folder(input_directory: str | Path, recursive: bool = False) -> list[Path]:
    """Return supported media files in deterministic, case-insensitive order."""
    root = Path(input_directory).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {root}")

    candidates = root.rglob("*") if recursive else root.iterdir()
    media_files = [
        path.resolve()
        for path in candidates
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(media_files, key=lambda path: path.as_posix().casefold())


def _optional_float(value: Any) -> float | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value in (None, "", "N/A"):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _string_tags(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {str(key): str(tag_value) for key, tag_value in value.items()}


def _get_tag(tags: dict[str, str], name: str) -> str | None:
    target = name.casefold()
    for key, value in tags.items():
        if key.casefold() == target:
            return value
    return None


def _extract_timecode(
    streams: list[dict[str, Any]],
    format_tags: dict[str, str],
) -> TimecodeMetadata:
    prioritized: list[tuple[int, int, TimecodeCandidate]] = []
    order = 0

    for stream in streams:
        stream_index = _optional_int(stream.get("index"))
        index_label = stream_index if stream_index is not None else "unknown"
        codec_type = str(stream.get("codec_type") or "unknown")
        codec_tag = str(stream.get("codec_tag_string") or "").casefold()
        tags = _string_tags(stream.get("tags"))
        value = _get_tag(tags, "timecode")

        is_timecode_track = codec_tag == "tmcd" or "timecode" in (
            _get_tag(tags, "handler_name") or ""
        ).casefold()
        priority = 0 if is_timecode_track else 1 if codec_type == "video" else 3

        if value:
            prioritized.append(
                (
                    priority,
                    order,
                    TimecodeCandidate(value=value, source=f"stream:{index_label}:{codec_type}"),
                )
            )
            order += 1

        side_data = stream.get("side_data_list")
        if isinstance(side_data, list):
            for side_data_index, item in enumerate(side_data):
                if not isinstance(item, dict):
                    continue
                side_value = item.get("timecode")
                if side_value:
                    prioritized.append(
                        (
                            priority,
                            order,
                            TimecodeCandidate(
                                value=str(side_value),
                                source=f"stream:{index_label}:side_data:{side_data_index}",
                            ),
                        )
                    )
                    order += 1

    format_timecode = _get_tag(format_tags, "timecode")
    if format_timecode:
        prioritized.append(
            (2, order, TimecodeCandidate(value=format_timecode, source="format:tags"))
        )

    prioritized.sort(key=lambda item: (item[0], item[1]))
    candidates: list[TimecodeCandidate] = []
    seen: set[tuple[str, str]] = set()
    for _, _, candidate in prioritized:
        key = (candidate.value, candidate.source)
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)

    selected = next(
        (candidate for candidate in candidates if SMPTE_TIMECODE_PATTERN.fullmatch(candidate.value)),
        None,
    )

    bwf_value = _get_tag(format_tags, "time_reference")
    if bwf_value is None:
        for stream in streams:
            bwf_value = _get_tag(_string_tags(stream.get("tags")), "time_reference")
            if bwf_value is not None:
                break

    return TimecodeMetadata(
        value=selected.value if selected else None,
        source=selected.source if selected else None,
        drop_frame=";" in selected.value if selected else None,
        candidates=candidates,
        bwf_time_reference_samples=_optional_int(bwf_value),
    )


def probe_media(media_path: str | Path) -> MediaProbe:
    path = Path(media_path).expanduser().resolve()
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    data = json.loads(run_command(command))
    streams = [stream for stream in data.get("streams", []) if isinstance(stream, dict)]
    format_data = data.get("format") if isinstance(data.get("format"), dict) else {}
    format_tags = _string_tags(format_data.get("tags"))

    video_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "video"),
        None,
    )
    audio_stream = next(
        (stream for stream in streams if stream.get("codec_type") == "audio"),
        None,
    )

    video = None
    if video_stream is not None:
        video = VideoStreamMetadata(
            index=_optional_int(video_stream.get("index")) or 0,
            codec_name=video_stream.get("codec_name"),
            width=_optional_int(video_stream.get("width")),
            height=_optional_int(video_stream.get("height")),
            frame_rate=video_stream.get("r_frame_rate"),
            average_frame_rate=video_stream.get("avg_frame_rate"),
            time_base=video_stream.get("time_base"),
            start_time_seconds=_optional_float(video_stream.get("start_time")),
        )

    audio = None
    if audio_stream is not None:
        audio = AudioStreamMetadata(
            index=_optional_int(audio_stream.get("index")) or 0,
            codec_name=audio_stream.get("codec_name"),
            sample_rate=_optional_int(audio_stream.get("sample_rate")),
            channels=_optional_int(audio_stream.get("channels")),
            channel_layout=audio_stream.get("channel_layout"),
            time_base=audio_stream.get("time_base"),
            start_time_seconds=_optional_float(audio_stream.get("start_time")),
        )

    return MediaProbe(
        format_name=format_data.get("format_name"),
        duration_seconds=_optional_float(format_data.get("duration")),
        start_time_seconds=_optional_float(format_data.get("start_time")),
        video=video,
        audio=audio,
        timecode=_extract_timecode(streams, format_tags),
        format_tags=format_tags,
    )


def _normalized_audio_path(
    source_path: Path,
    input_root: Path,
    audio_output_root: Path,
) -> Path:
    relative_path = source_path.relative_to(input_root)
    # Keep the source extension in the output name so camera.mov and camera.wav
    # never overwrite each other.
    output_relative_path = relative_path.with_suffix(relative_path.suffix + ".wav")
    return audio_output_root / output_relative_path


def save_manifest(manifest: IngestManifest, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def ingest_folder(
    input_directory: str | Path,
    output_directory: str | Path,
    manifest_path: str | Path | None = None,
    recursive: bool = False,
) -> tuple[IngestManifest, Path]:
    """Probe all supported media and create normalized audio plus a JSON manifest."""
    if not check_ffmpeg():
        raise RuntimeError("FFmpeg and ffprobe must both be installed and available on PATH.")

    input_root = Path(input_directory).expanduser().resolve()
    output_root = Path(output_directory).expanduser().resolve()
    if output_root == input_root:
        raise ValueError("Output directory must be different from the input directory.")

    media_files = scan_input_folder(input_root, recursive=recursive)
    if output_root.is_relative_to(input_root):
        media_files = [path for path in media_files if not path.is_relative_to(output_root)]
    if not media_files:
        raise ValueError(f"No supported .mp4, .mov, or .wav files found in {input_root}")

    audio_output_root = output_root / "audio"
    assets: list[IngestedAsset] = []

    for index, source_path in enumerate(media_files, start=1):
        probe = probe_media(source_path)
        if probe.audio is None:
            raise ValueError(f"Media file has no audio stream: {source_path}")

        normalized_audio = _normalized_audio_path(
            source_path=source_path,
            input_root=input_root,
            audio_output_root=audio_output_root,
        )
        extract_audio(
            input_video=str(source_path),
            output_audio=str(normalized_audio),
            sample_rate=16_000,
            channels=1,
        )

        media_kind = "video" if source_path.suffix.lower() in VIDEO_EXTENSIONS else "audio"
        assets.append(
            IngestedAsset(
                asset_id=f"asset_{index:04d}",
                media_kind=media_kind,
                source_path=str(source_path),
                source_relative_path=source_path.relative_to(input_root).as_posix(),
                source_extension=source_path.suffix.lower(),
                file_size_bytes=source_path.stat().st_size,
                normalized_audio_path=str(normalized_audio.resolve()),
                probe=probe,
            )
        )

    manifest = IngestManifest(
        schema_version="1.0",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        input_directory=str(input_root),
        output_directory=str(output_root),
        audio_spec={
            "sample_rate_hz": 16_000,
            "channels": 1,
            "sample_format": "signed 16-bit PCM little-endian",
            "codec": "pcm_s16le",
        },
        assets=assets,
    )

    resolved_manifest_path = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else output_root / "ingest_manifest.json"
    )
    return manifest, save_manifest(manifest, resolved_manifest_path)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest podcast media and create normalized audio plus a metadata manifest."
    )
    parser.add_argument("input_directory", help="Folder containing .mp4, .mov, and .wav files")
    parser.add_argument(
        "--output-directory",
        default="output/ingest",
        help="Destination for normalized audio and ingest_manifest.json",
    )
    parser.add_argument("--manifest-path", default=None, help="Optional manifest JSON path")
    parser.add_argument("--recursive", action="store_true", help="Scan nested directories")
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    manifest, path = ingest_folder(
        input_directory=args.input_directory,
        output_directory=args.output_directory,
        manifest_path=args.manifest_path,
        recursive=args.recursive,
    )
    print(f"Ingested {len(manifest.assets)} media file(s).")
    print(f"Manifest: {path}")
    missing_timecode = [
        asset.source_relative_path
        for asset in manifest.assets
        if asset.media_kind == "video" and asset.probe.timecode.value is None
    ]
    if missing_timecode:
        print("Warning: no SMPTE timecode metadata found in:")
        for source_path in missing_timecode:
            print(f"  - {source_path}")


if __name__ == "__main__":
    main()
