from dataclasses import dataclass

from src.analysis.schemas import AnalyzedScene


@dataclass
class ScoredScene:
    start: float
    end: float
    text: str
    score: float
    reason: str


KEYWORDS = [
    "important",
    "key",
    "problem",
    "solution",
    "demo",
    "example",
    "result",
    "tip",
    "first",
    "next",
    "finally",
    "quan trọng",
    "vấn đề",
    "giải pháp",
    "ví dụ",
    "kết quả",
    "mẹo",
]


def score_scene(scene: AnalyzedScene) -> ScoredScene:
    score = 0.0
    reasons = []

    duration = scene.end - scene.start
    text = scene.text.strip()
    lower_text = text.lower()

    if 3 <= duration <= 20:
        score += 30
        reasons.append("duration tốt")

    if len(text) >= 30:
        score += 30
        reasons.append("có lời thoại")

    matched_keywords = [kw for kw in KEYWORDS if kw in lower_text]
    if matched_keywords:
        score += 30
        reasons.append(f"có keyword: {', '.join(matched_keywords)}")

    if text.endswith(("!", "?")):
        score += 10
        reasons.append("câu có cảm xúc/câu hỏi")

    return ScoredScene(
        start=scene.start,
        end=scene.end,
        text=scene.text,
        score=min(score, 100),
        reason=", ".join(reasons) if reasons else "ít tín hiệu nổi bật",
    )


def score_scenes(scenes: list[AnalyzedScene]) -> list[ScoredScene]:
    scored = [score_scene(scene) for scene in scenes]
    return sorted(scored, key=lambda item: item.score, reverse=True)