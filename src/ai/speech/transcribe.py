from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from src.ai.speech.schemas import SpeechSegment, TranscriptWord


class SpeechTranscriber:
    """Local Faster Whisper transcription backend."""

    def __init__(
        self,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str | None = None,
    ):
        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )
        self.language = language

    def transcribe(self, audio_path: str) -> list[SpeechSegment]:
        """Return segment-level output for compatibility with the existing pipeline."""
        segments, _ = self.model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            language=self.language,
        )

        results: list[SpeechSegment] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            results.append(
                SpeechSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=text,
                )
            )
        return results

    def transcribe_words(
        self,
        audio_path: str | Path,
        speaker_id: str,
    ) -> list[TranscriptWord]:
        """Transcribe one isolated audio stem into the Module 2 word schema."""
        normalized_speaker_id = speaker_id.strip()
        if not normalized_speaker_id:
            raise ValueError("speaker_id must not be empty")

        path = Path(audio_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {path}")

        segments, _ = self.model.transcribe(
            str(path),
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
            language=self.language,
        )

        results: list[TranscriptWord] = []
        for segment in segments:
            words = getattr(segment, "words", None)
            if words is None:
                if segment.text.strip():
                    raise RuntimeError(
                        "Whisper returned text without word timestamps. "
                        "Verify that the installed Faster Whisper version supports "
                        "word_timestamps=True."
                    )
                continue

            for word_info in words:
                word = str(word_info.word).strip()
                if not word:
                    continue
                if word_info.start is None or word_info.end is None:
                    raise RuntimeError(f"Missing timestamp for transcribed word: {word!r}")

                start_ms = max(0, round(float(word_info.start) * 1000))
                end_ms = max(start_ms, round(float(word_info.end) * 1000))
                results.append(
                    TranscriptWord(
                        word=word,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        speaker_id=normalized_speaker_id,
                    )
                )

        return results


def save_transcript_json(
    segments: list[SpeechSegment],
    output_path: str | Path,
) -> Path:
    """Save the legacy segment-level transcript used by the current prototype."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {
            "start": segment.start,
            "end": segment.end,
            "text": segment.text,
        }
        for segment in segments
    ]
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def save_word_transcript_json(
    words: list[TranscriptWord],
    output_path: str | Path,
) -> Path:
    """Save the strict Module 2 JSON array contract."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(word) for word in words], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _safe_speaker_id(value: str) -> str:
    speaker_id = re.sub(r"[^\w.-]+", "_", value.strip(), flags=re.UNICODE).strip("_.-")
    return speaker_id or "unknown"


def _load_speaker_map(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}

    speaker_map_path = Path(path).expanduser().resolve()
    data = json.loads(speaker_map_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Speaker map must be a JSON object of asset/path keys to speaker IDs.")

    result: dict[str, str] = {}
    for key, value in data.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid speaker ID for speaker map key: {key!r}")
        result[str(key)] = value.strip()
    return result


def _speaker_id_for_asset(
    asset: dict[str, Any],
    speaker_map: dict[str, str],
    selected_asset_count: int,
) -> str:
    source_relative_path = str(asset.get("source_relative_path") or "")
    source_path = str(asset.get("source_path") or "")
    normalized_audio_path = str(asset.get("normalized_audio_path") or "")
    lookup_keys = [
        str(asset.get("asset_id") or ""),
        source_relative_path,
        Path(source_relative_path).name if source_relative_path else "",
        source_path,
        normalized_audio_path,
    ]
    for key in lookup_keys:
        if key and key in speaker_map:
            return speaker_map[key]

    if asset.get("media_kind") == "audio" and source_relative_path:
        return _safe_speaker_id(Path(source_relative_path).stem)

    if selected_asset_count == 1:
        return "unknown"

    raise ValueError(
        "speaker_id cannot be inferred for multiple video audio tracks. "
        "Provide --speaker-map or select isolated WAV assets."
    )


def _select_manifest_assets(
    manifest: dict[str, Any],
    asset_ids: set[str] | None,
) -> list[dict[str, Any]]:
    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, list):
        raise ValueError("Invalid ingest manifest: assets must be an array.")
    assets = [asset for asset in raw_assets if isinstance(asset, dict)]

    if asset_ids:
        selected = [asset for asset in assets if str(asset.get("asset_id")) in asset_ids]
        found_ids = {str(asset.get("asset_id")) for asset in selected}
        missing_ids = sorted(asset_ids - found_ids)
        if missing_ids:
            raise ValueError(f"Asset IDs not found in manifest: {', '.join(missing_ids)}")
        return selected

    # Isolated WAV stems are authoritative. Camera scratch audio is only used as
    # a fallback when the manifest contains exactly one media asset.
    audio_assets = [asset for asset in assets if asset.get("media_kind") == "audio"]
    if audio_assets:
        return audio_assets
    if len(assets) == 1:
        return assets
    if not assets:
        raise ValueError("Ingest manifest contains no media assets.")
    raise ValueError(
        "Manifest has multiple media assets but no isolated WAV stems. "
        "Select assets explicitly with --asset-id and provide --speaker-map."
    )


def transcribe_ingest_manifest(
    manifest_path: str | Path,
    transcriber: SpeechTranscriber,
    speaker_map_path: str | Path | None = None,
    asset_ids: list[str] | None = None,
) -> list[TranscriptWord]:
    """Transcribe normalized audio referenced by a Module 1 manifest."""
    path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Invalid ingest manifest: root must be a JSON object.")

    selected_assets = _select_manifest_assets(
        manifest,
        set(asset_ids) if asset_ids else None,
    )
    speaker_map = _load_speaker_map(speaker_map_path)
    all_words: list[TranscriptWord] = []

    for asset in selected_assets:
        audio_path = asset.get("normalized_audio_path")
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError(
                f"Asset {asset.get('asset_id', '<unknown>')} has no normalized_audio_path."
            )
        speaker_id = _speaker_id_for_asset(asset, speaker_map, len(selected_assets))
        all_words.extend(transcriber.transcribe_words(audio_path, speaker_id))

    return sorted(
        all_words,
        key=lambda item: (item.start_ms, item.end_ms, item.speaker_id, item.word),
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default="small", help="Faster Whisper model name or path")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or auto")
    parser.add_argument("--compute-type", default="int8", help="Faster Whisper compute type")
    parser.add_argument("--language", default=None, help="Optional ISO language code, e.g. en or vi")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate word-level podcast transcripts with speaker IDs."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audio_parser = subparsers.add_parser("audio", help="Transcribe one audio file")
    audio_parser.add_argument("audio_path")
    audio_parser.add_argument("--speaker-id", required=True)
    audio_parser.add_argument("--output", default="output/transcript_words.json")
    _add_model_arguments(audio_parser)

    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Transcribe isolated WAV stems from a Module 1 ingest manifest",
    )
    manifest_parser.add_argument("manifest_path")
    manifest_parser.add_argument("--speaker-map", default=None)
    manifest_parser.add_argument(
        "--asset-id",
        action="append",
        default=None,
        help="Asset ID to transcribe; repeat to select multiple assets",
    )
    manifest_parser.add_argument("--output", default="output/transcript_words.json")
    _add_model_arguments(manifest_parser)
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    transcriber = SpeechTranscriber(
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
    )

    if args.command == "audio":
        words = transcriber.transcribe_words(args.audio_path, args.speaker_id)
    else:
        words = transcribe_ingest_manifest(
            manifest_path=args.manifest_path,
            transcriber=transcriber,
            speaker_map_path=args.speaker_map,
            asset_ids=args.asset_id,
        )

    output_path = save_word_transcript_json(words, args.output)
    print(f"Transcribed {len(words)} word(s).")
    print(f"Transcript: {output_path}")


if __name__ == "__main__":
    main()
