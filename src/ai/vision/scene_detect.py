import json
from pathlib import Path

from scenedetect import detect, ContentDetector

from src.ai.vision.schemas import SceneSegment


def detect_scenes(
    video_path: str,
    threshold: float = 27.0,
) -> list[SceneSegment]:
    scene_list = detect(
        video_path,
        ContentDetector(threshold=threshold),
    )

    scenes: list[SceneSegment] = []

    for start_time, end_time in scene_list:
        scenes.append(
            SceneSegment(
                start=start_time.get_seconds(),
                end=end_time.get_seconds(),
            )
        )

    return scenes


def save_scenes_json(
    scenes: list[SceneSegment],
    output_path: str,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "start": scene.start,
            "end": scene.end,
        }
        for scene in scenes
    ]

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return path