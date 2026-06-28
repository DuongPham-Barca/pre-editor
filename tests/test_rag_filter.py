import json
import tempfile
import unittest
from pathlib import Path

import httpx

from src.ai.speech.schemas import TranscriptWord
from src.analysis.rag_filter import (
    LLMRequestError,
    OpenAICompatibleJsonClient,
    RagNarrativeFilter,
    build_transcript_blocks,
    load_transcript_words,
    merge_keep_zones,
    save_keep_zones_json,
)
from src.analysis.schemas import KeepZone


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def generate_json(self, system_prompt, user_prompt, json_schema):
        self.prompts.append((system_prompt, user_prompt, json_schema))
        return next(self.responses)


class TranscriptLoadingTests(unittest.TestCase):
    def test_loads_and_sorts_word_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcript.json"
            path.write_text(
                json.dumps(
                    [
                        {"word": "second", "start_ms": 200, "end_ms": 300, "speaker_id": "B"},
                        {"word": "first", "start_ms": 10, "end_ms": 100, "speaker_id": "A"},
                    ]
                ),
                encoding="utf-8",
            )
            words = load_transcript_words(path)

        self.assertEqual([word.word for word in words], ["first", "second"])

    def test_rejects_segment_level_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcript.json"
            path.write_text(
                json.dumps([{"text": "legacy", "start": 0.0, "end": 1.0}]),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_transcript_words(path)


class TranscriptBlockTests(unittest.TestCase):
    def test_splits_blocks_on_speaker_change(self) -> None:
        words = [
            TranscriptWord("Hello", 0, 100, "Host"),
            TranscriptWord("there", 110, 300, "Host"),
            TranscriptWord("Hi", 320, 500, "Guest"),
        ]
        blocks = build_transcript_blocks(words)

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].text, "Hello there")
        self.assertEqual(blocks[1].speaker_id, "Guest")

    def test_splits_blocks_at_sentence_boundaries(self) -> None:
        words = [
            TranscriptWord("First", 0, 100, "Host"),
            TranscriptWord("sentence.", 110, 300, "Host"),
            TranscriptWord("Second", 320, 500, "Host"),
        ]
        blocks = build_transcript_blocks(words)

        self.assertEqual([block.text for block in blocks], ["First sentence.", "Second"])


class NarrativeFilterTests(unittest.TestCase):
    def test_filters_and_merges_overlapping_keep_zones(self) -> None:
        words = [
            TranscriptWord("Intro.", 0, 500, "Host"),
            TranscriptWord("problem.", 600, 1_000, "Host"),
            TranscriptWord("solution.", 1_100, 1_500, "Host"),
        ]
        client = FakeLLMClient(
            [
                {
                    "keep_zones": [
                        {"start_ms": 0, "end_ms": 1_000, "topic": "Problem"},
                        {"start_ms": 600, "end_ms": 1_500, "topic": "Solution"},
                    ]
                }
            ]
        )

        zones = RagNarrativeFilter(client).filter("Discuss problem and solution", words)

        self.assertEqual(zones, [KeepZone(0, 1_500, "Problem / Solution")])
        self.assertIn("PODCAST PLAN", client.prompts[0][1])
        self.assertIn("[0-500] Host", client.prompts[0][1])

    def test_rejects_llm_timestamp_outside_chunk(self) -> None:
        words = [TranscriptWord("Hello", 1_000, 1_500, "Host")]
        client = FakeLLMClient(
            [{"keep_zones": [{"start_ms": 0, "end_ms": 1_500, "topic": "Intro"}]}]
        )

        with self.assertRaises(LLMRequestError):
            RagNarrativeFilter(client).filter("Introduction", words)

    def test_rejects_timestamp_that_is_not_a_transcript_boundary(self) -> None:
        words = [
            TranscriptWord("Hello", 1_000, 1_200, "Host"),
            TranscriptWord("world.", 1_300, 1_500, "Host"),
        ]
        client = FakeLLMClient(
            [{"keep_zones": [{"start_ms": 1_100, "end_ms": 1_500, "topic": "Intro"}]}]
        )

        with self.assertRaises(LLMRequestError):
            RagNarrativeFilter(client).filter("Introduction", words)

    def test_empty_transcript_does_not_call_llm(self) -> None:
        client = FakeLLMClient([])
        zones = RagNarrativeFilter(client).filter("Plan", [])
        self.assertEqual(zones, [])
        self.assertEqual(client.prompts, [])


class ZoneSerializationTests(unittest.TestCase):
    def test_saves_keep_zones_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zones.json"
            save_keep_zones_json([KeepZone(10, 500, "Introduction")], path)
            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            data,
            {"keep_zones": [{"start_ms": 10, "end_ms": 500, "topic": "Introduction"}]},
        )

    def test_merges_adjacent_same_topic_only(self) -> None:
        result = merge_keep_zones(
            [
                KeepZone(0, 1_000, "Intro"),
                KeepZone(1_500, 2_000, "Intro"),
                KeepZone(4_000, 5_000, "Other"),
            ]
        )
        self.assertEqual(
            result,
            [KeepZone(0, 2_000, "Intro"), KeepZone(4_000, 5_000, "Other")],
        )


class HttpClientTests(unittest.TestCase):
    def test_sends_structured_schema_and_parses_json_content(self) -> None:
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["authorization"] = request.headers.get("Authorization")
            captured["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "keep_zones": [
                                            {"start_ms": 0, "end_ms": 100, "topic": "Intro"}
                                        ]
                                    }
                                )
                            }
                        }
                    ]
                },
            )

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        client = OpenAICompatibleJsonClient(
            endpoint="https://llm.example.test/chat/completions",
            model="test-model",
            api_key="secret",
            client=http_client,
        )
        result = client.generate_json("system", "user", {"type": "object"})

        self.assertIn("keep_zones", result)
        self.assertEqual(captured["authorization"], "Bearer secret")
        self.assertEqual(
            captured["payload"]["response_format"]["type"],
            "json_schema",
        )
        http_client.close()


if __name__ == "__main__":
    unittest.main()
