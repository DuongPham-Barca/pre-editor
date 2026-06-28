from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom

from src.media.metadata import VideoMetadata
from src.timeline.schemas import TimelineClip


def _format_name(width: int, height: int, fps: float) -> str:
    res = "1080" if height <= 1080 else "2160" if height <= 2160 else "4320"
    if abs(fps - 23.976) < 0.005:
        fr = "2398"
    elif abs(fps - 24) < 0.005:
        fr = "24"
    elif abs(fps - 25) < 0.005:
        fr = "25"
    elif abs(fps - 29.97) < 0.005:
        fr = "2997"
    elif abs(fps - 30) < 0.005:
        fr = "30"
    elif abs(fps - 50) < 0.005:
        fr = "50"
    elif abs(fps - 59.94) < 0.005:
        fr = "5994"
    elif abs(fps - 60) < 0.005:
        fr = "60"
    else:
        fr = str(int(fps))
    return f"FFVideoFormat{res}{fr}"


def _frame_duration(fps: float) -> str:
    if abs(fps - 23.976) < 0.005:
        return "1001/24000s"
    if abs(fps - 29.97) < 0.005:
        return "1001/30000s"
    if abs(fps - 59.94) < 0.005:
        return "1001/60000s"
    return f"1/{round(fps)}s"


def _time_str(seconds: float, fps: float) -> str:
    frames = round(seconds * fps)
    return f"{frames}/{fps}s"


def _frames_to_str(frames: int, fps: float) -> str:
    return f"{frames}/{fps}s"


def _sanitise_name(text: str, max_len: int = 50) -> str:
    clean = re.sub(r"[^\w\s-]", "", text).strip()
    if len(clean) > max_len:
        clean = clean[:max_len].rstrip()
    return clean or "clip"


def export_fcpxml(
    timeline: list[TimelineClip],
    video_path: str | Path,
    metadata: VideoMetadata,
    output_path: str | Path,
    safety_handle_s: float = 0.5,
    project_name: str = "Pre-Editor Output",
) -> Path:
    video_path = Path(video_path).resolve()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fps = metadata.fps
    fmt_name = _format_name(metadata.width, metadata.height, fps)
    fmt_duration = _frame_duration(fps)

    root = ET.Element("fcpxml", version="1.8")

    # ── Resources ──────────────────────────────────────────
    resources = EL_SubElement(root, "resources")

    format_id = "r1"
    EL_SubElement(
        resources, "format",
        id=format_id,
        name=fmt_name,
        frameDuration=fmt_duration,
        width=str(metadata.width),
        height=str(metadata.height),
    )

    asset_id = "r2"
    EL_SubElement(
        resources, "asset",
        id=asset_id,
        name=video_path.name,
        src=video_path.as_uri(),
        format=format_id,
        hasVideo="1",
        hasAudio="1",
    )

    # ── Project / Sequence / Spine ─────────────────────────
    project = EL_SubElement(root, "project", name=project_name)
    sequence = EL_SubElement(project, "sequence")
    spine = EL_SubElement(sequence, "spine")

    offset_frames = 0

    for i, clip in enumerate(timeline):
        original_duration = clip.source_end - clip.source_start
        xml_start = max(0.0, clip.source_start - safety_handle_s)
        xml_duration = original_duration + 2.0 * safety_handle_s

        dur_frames = round(xml_duration * fps)
        start_frames = round(xml_start * fps)

        EL_SubElement(
            spine, "asset-clip",
            ref=asset_id,
            offset=_frames_to_str(offset_frames, fps),
            duration=_frames_to_str(dur_frames, fps),
            start=_frames_to_str(start_frames, fps),
            name=f"Clip {i+1}: {_sanitise_name(clip.text, 40)}",
            audioRole="dialogue",
            videoRole="video",
        )

        offset_frames += dur_frames

    # ── Serialise ──────────────────────────────────────────
    raw = ET.tostring(root, encoding="unicode")
    dom = minidom.parseString(raw)
    pretty = dom.toprettyxml(indent="  ")

    pretty = pretty.replace(
        '<?xml version="1.0" ?>',
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>',
    )

    output_path.write_text(pretty, encoding="utf-8")
    return output_path


def EL_SubElement(parent: ET.Element, tag: str, **attrs: str) -> ET.Element:
    el = ET.SubElement(parent, tag)
    for k, v in attrs.items():
        el.set(k.replace("_", "-"), v)
    return el
