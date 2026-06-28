import json
from pathlib import Path

from src.analysis.scoring.highlight import ScoredScene
from src.analysis.schemas import KeepZone
from src.timeline.schemas import TimelineClip


def build_timeline(
    scored_scenes: list[ScoredScene],
    min_score: float = 60,
    max_clips: int = 10,
) -> list[TimelineClip]:
    selected = [
        scene for scene in scored_scenes
        if scene.score >= min_score
    ][:max_clips]

    selected = sorted(selected, key=lambda scene: scene.start)

    timeline: list[TimelineClip] = []
    current_time = 0.0

    for scene in selected:
        duration = scene.end - scene.start

        clip = TimelineClip(
            source_start=scene.start,
            source_end=scene.end,
            timeline_start=current_time,
            timeline_end=current_time + duration,
            text=scene.text,
            score=scene.score,
        )

        timeline.append(clip)
        current_time += duration

    return timeline


def save_timeline_json(
    timeline: list[TimelineClip],
    output_path: str,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "source_start": clip.source_start,
            "source_end": clip.source_end,
            "timeline_start": clip.timeline_start,
            "timeline_end": clip.timeline_end,
            "text": clip.text,
            "score": clip.score,
        }
        for clip in timeline
    ]

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return path


def build_timeline_from_zones(
    zones: list[KeepZone],
) -> list[TimelineClip]:
    timeline: list[TimelineClip] = []
    current_time = 0.0

    for zone in zones:
        duration_s = (zone.end_ms - zone.start_ms) / 1000.0
        timeline.append(
            TimelineClip(
                source_start=zone.start_ms / 1000.0,
                source_end=zone.end_ms / 1000.0,
                timeline_start=current_time,
                timeline_end=current_time + duration_s,
                text=zone.topic,
                score=100.0,
            )
        )
        current_time += duration_s

    return timeline