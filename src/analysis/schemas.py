from dataclasses import dataclass


@dataclass
class AnalyzedScene:
    start: float
    end: float
    text: str