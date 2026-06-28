from src.ai.speech.schemas import SpeechSegment
from src.ai.vision.schemas import SceneSegment
from src.analysis.schemas import AnalyzedScene


def merge_scenes_with_transcript(
    scenes: list[SceneSegment],
    transcript: list[SpeechSegment],
) -> list[AnalyzedScene]:
    analyzed: list[AnalyzedScene] = []

    for scene in scenes:
        texts: list[str] = []

        for segment in transcript:
            is_inside_scene = (
                segment.start < scene.end
                and segment.end > scene.start
            )

            if is_inside_scene:
                texts.append(segment.text)

        analyzed.append(
            AnalyzedScene(
                start=scene.start,
                end=scene.end,
                text=" ".join(texts).strip(),
            )
        )

    return analyzed