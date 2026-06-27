from dataclasses import dataclass


@dataclass
class TimelineClip:
    source_start: float
    source_end: float
    timeline_start: float
    timeline_end: float
    text: str
    score: float