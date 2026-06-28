import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.ai.speech.schemas import TranscriptWord
from src.ai.speech.transcribe import (
    SpeechTranscriber,
    save_word_transcript_json,
    transcribe_ingest_manifest,
)


class SpeechTranscriberTests(unittest.TestCase):
    @patch("src.ai.speech.transcribe.WhisperModel")
    def test_emits_word_level_milliseconds_and_speaker_id(self, model_class) -> None:
        word_items = [
            SimpleNamespace(word=" Hello", start=0.1234, end=0.5016),
            SimpleNamespace(word=" world", start=0.51, end=1.009),
        ]
        segment = SimpleNamespace(text=" Hello world", words=word_items)
        model_class.return_value.transcribe.return_value = ([segment], object())

        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "mic.wav"
            audio_path.touch()
            transcriber = SpeechTranscriber()
            result = transcriber.transcribe_words(audio_path, "Mic_1")

        self.assertEqual(
            result,
            [
                TranscriptWord("Hello", 123, 502, "Mic_1"),
                TranscriptWord("world", 510, 1009, "Mic_1"),
            ],
        )
        call_options = model_class.return_value.transcribe.call_args.kwargs
        self.assertTrue(call_options["word_timestamps"])
        self.assertTrue(call_options["vad_filter"])

    @patch("src.ai.speech.transcribe.WhisperModel")
    def test_rejects_text_without_word_timestamps(self, model_class) -> None:
        segment = SimpleNamespace(text="Speech exists", words=None)
        model_class.return_value.transcribe.return_value = ([segment], object())

        with tempfile.TemporaryDirectory() as directory:
            audio_path = Path(directory) / "mic.wav"
            audio_path.touch()
            transcriber = SpeechTranscriber()
            with self.assertRaises(RuntimeError):
                transcriber.transcribe_words(audio_path, "Mic_1")


class TranscriptSerializationTests(unittest.TestCase):
    def test_saves_strict_json_array_contract(self) -> None:
        words = [TranscriptWord("Hello", 10, 200, "Host")]
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "transcript.json"
            save_word_transcript_json(words, output_path)
            data = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(
            data,
            [{"word": "Hello", "start_ms": 10, "end_ms": 200, "speaker_id": "Host"}],
        )


class ManifestTranscriptionTests(unittest.TestCase):
    def test_transcribes_audio_stems_and_sorts_combined_words(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            host_audio = root / "host.wav"
            guest_audio = root / "guest.wav"
            host_audio.touch()
            guest_audio.touch()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "asset_0001",
                                "media_kind": "video",
                                "source_relative_path": "camera.mov",
                                "normalized_audio_path": str(root / "camera.wav"),
                            },
                            {
                                "asset_id": "asset_0002",
                                "media_kind": "audio",
                                "source_relative_path": "Host Mic.wav",
                                "normalized_audio_path": str(host_audio),
                            },
                            {
                                "asset_id": "asset_0003",
                                "media_kind": "audio",
                                "source_relative_path": "Guest Mic.wav",
                                "normalized_audio_path": str(guest_audio),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            transcriber = MagicMock()
            transcriber.transcribe_words.side_effect = [
                [TranscriptWord("later", 500, 700, "Host_Mic")],
                [TranscriptWord("first", 100, 300, "Guest_Mic")],
            ]

            result = transcribe_ingest_manifest(manifest_path, transcriber)

        self.assertEqual([word.word for word in result], ["first", "later"])
        self.assertEqual(transcriber.transcribe_words.call_count, 2)
        speaker_ids = {
            call.args[1] for call in transcriber.transcribe_words.call_args_list
        }
        self.assertEqual(speaker_ids, {"Host_Mic", "Guest_Mic"})

    def test_speaker_map_overrides_inferred_stem_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_path = root / "mic.wav"
            audio_path.touch()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "assets": [
                            {
                                "asset_id": "asset_0001",
                                "media_kind": "audio",
                                "source_relative_path": "mic.wav",
                                "normalized_audio_path": str(audio_path),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            speaker_map_path = root / "speakers.json"
            speaker_map_path.write_text(
                json.dumps({"asset_0001": "Django"}),
                encoding="utf-8",
            )
            transcriber = MagicMock()
            transcriber.transcribe_words.return_value = []

            transcribe_ingest_manifest(
                manifest_path,
                transcriber,
                speaker_map_path=speaker_map_path,
            )

        self.assertEqual(transcriber.transcribe_words.call_args.args[1], "Django")


if __name__ == "__main__":
    unittest.main()
