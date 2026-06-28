from dataclasses import dataclass


@dataclass
class SpeechSegment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptWord:
    word: str
    start_ms: int
    end_ms: int
    speaker_id: str
