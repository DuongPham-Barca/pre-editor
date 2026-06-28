from dataclasses import dataclass


@dataclass
class AnalyzedScene:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class KeepZone:
    start_ms: int
    end_ms: int
    topic: str
