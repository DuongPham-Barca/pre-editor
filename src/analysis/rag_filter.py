from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx
from dotenv import load_dotenv

from src.ai.llm.huggingface import (
    DEFAULT_HF_FILENAME,
    DEFAULT_HF_REPO,
    HuggingFaceGGUFJsonClient,
    download_gguf_model,
)
from src.ai.speech.schemas import TranscriptWord
from src.analysis.schemas import KeepZone


KEEP_ZONES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "keep_zones": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_ms": {"type": "integer", "minimum": 0},
                    "end_ms": {"type": "integer", "minimum": 1},
                    "topic": {"type": "string", "minLength": 1},
                },
                "required": ["start_ms", "end_ms", "topic"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["keep_zones"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You are a senior long-form podcast story editor.
Select only transcript ranges that materially support the supplied Podcast Plan.
Preserve chronological order and coherent conversational context.
Exclude off-topic tangents, repetition, housekeeping, and empty chatter.
Never invent timestamps. Every start_ms and end_ms must be inside the supplied
chunk window and must correspond to content visible in the transcript.
Return only JSON matching the supplied keep_zones schema."""


class JsonLLMClient(Protocol):
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> dict[str, Any]: ...


class LLMRequestError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscriptBlock:
    start_ms: int
    end_ms: int
    speaker_id: str
    text: str


@dataclass(frozen=True)
class TranscriptChunk:
    start_ms: int
    end_ms: int
    text: str
    block_starts: tuple[int, ...]
    block_ends: tuple[int, ...]


class OpenAICompatibleJsonClient:
    """Minimal client for chat-completions-compatible structured JSON endpoints."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
    ):
        if not endpoint.strip():
            raise ValueError("LLM endpoint must not be empty")
        if not model.strip():
            raise ValueError("LLM model must not be empty")
        self.endpoint = endpoint
        self.model = model
        self.api_key = api_key
        self.max_retries = max(0, max_retries)
        self._client = client or httpx.Client(timeout=timeout_seconds)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenAICompatibleJsonClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",  
                "json_schema": {
                    "name": "podcast_keep_zones",
                    "strict": True,
                    "schema": json_schema,
                },
            },
        }

        response: httpx.Response | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(self.endpoint, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                if attempt >= self.max_retries:
                    raise LLMRequestError(f"LLM request failed: {exc.__class__.__name__}") from exc
                time.sleep(2**attempt)
                continue

            if response.status_code < 400:
                break
            if response.status_code not in {408, 409, 429} and response.status_code < 500:
                raise LLMRequestError(f"LLM request failed with HTTP {response.status_code}")
            if attempt >= self.max_retries:
                raise LLMRequestError(f"LLM request failed with HTTP {response.status_code}")
            time.sleep(2**attempt)

        if response is None:
            raise LLMRequestError("LLM request returned no response")

        try:
            body = response.json()
            message = body["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMRequestError("LLM response has an unsupported response envelope") from exc

        parsed = message.get("parsed") if isinstance(message, dict) else None
        if isinstance(parsed, dict):
            return parsed

        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict)
            )
        if not isinstance(content, str):
            raise LLMRequestError("LLM response does not contain JSON text")
        return _parse_json_object(content)


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMRequestError("LLM returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise LLMRequestError("LLM JSON root must be an object")
    return value


def load_podcast_plan(path: str | Path) -> str:
    plan_path = Path(path).expanduser().resolve()
    plan = plan_path.read_text(encoding="utf-8").strip()
    if not plan:
        raise ValueError(f"Podcast Plan is empty: {plan_path}")
    return plan


def load_transcript_words(path: str | Path) -> list[TranscriptWord]:
    transcript_path = Path(path).expanduser().resolve()
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Transcript JSON root must be an array.")

    words: list[TranscriptWord] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Transcript item {index} must be an object.")
        word = item.get("word")
        start_ms = item.get("start_ms")
        end_ms = item.get("end_ms")
        speaker_id = item.get("speaker_id")
        if not isinstance(word, str) or not word.strip():
            raise ValueError(f"Transcript item {index} has an invalid word.")
        if not isinstance(start_ms, int) or isinstance(start_ms, bool) or start_ms < 0:
            raise ValueError(f"Transcript item {index} has an invalid start_ms.")
        if not isinstance(end_ms, int) or isinstance(end_ms, bool) or end_ms <= start_ms:
            raise ValueError(f"Transcript item {index} has an invalid end_ms.")
        if not isinstance(speaker_id, str) or not speaker_id.strip():
            raise ValueError(f"Transcript item {index} has an invalid speaker_id.")
        words.append(
            TranscriptWord(
                word=word.strip(),
                start_ms=start_ms,
                end_ms=end_ms,
                speaker_id=speaker_id.strip(),
            )
        )

    return sorted(words, key=lambda item: (item.start_ms, item.end_ms, item.speaker_id))


def build_transcript_blocks(
    words: list[TranscriptWord],
    max_block_duration_ms: int = 30_000,
    max_words_per_block: int = 80,
    speaker_gap_ms: int = 1_500,
) -> list[TranscriptBlock]:
    if not words:
        return []

    blocks: list[TranscriptBlock] = []
    current_words: list[TranscriptWord] = []

    def flush() -> None:
        if not current_words:
            return
        blocks.append(
            TranscriptBlock(
                start_ms=current_words[0].start_ms,
                end_ms=current_words[-1].end_ms,
                speaker_id=current_words[0].speaker_id,
                text=" ".join(item.word for item in current_words),
            )
        )
        current_words.clear()

    for word in words:
        if current_words:
            first = current_words[0]
            previous = current_words[-1]
            should_split = (
                word.speaker_id != first.speaker_id
                or word.start_ms - previous.end_ms > speaker_gap_ms
                or word.end_ms - first.start_ms > max_block_duration_ms
                or len(current_words) >= max_words_per_block
            )
            if should_split:
                flush()
        current_words.append(word)
        if word.word.rstrip().endswith((".", "?", "!")):
            flush()
    flush()
    return blocks


def _format_block(block: TranscriptBlock) -> str:
    return f"[{block.start_ms}-{block.end_ms}] {block.speaker_id}: {block.text}"


def chunk_transcript_blocks(
    blocks: list[TranscriptBlock],
    max_transcript_chars: int = 24_000,
    overlap_blocks: int = 1,
) -> list[TranscriptChunk]:
    if max_transcript_chars < 1_000:
        raise ValueError("max_transcript_chars must be at least 1000")
    if not blocks:
        return []

    if overlap_blocks < 0:
        raise ValueError("overlap_blocks must not be negative")

    chunks: list[TranscriptChunk] = []
    start_index = 0
    while start_index < len(blocks):
        end_index = start_index
        current_length = 0
        while end_index < len(blocks):
            line_length = len(_format_block(blocks[end_index])) + 1
            if end_index > start_index and current_length + line_length > max_transcript_chars:
                break
            current_length += line_length
            end_index += 1

        current = blocks[start_index:end_index]
        chunks.append(
            TranscriptChunk(
                start_ms=current[0].start_ms,
                end_ms=current[-1].end_ms,
                text="\n".join(_format_block(block) for block in current),
                block_starts=tuple(block.start_ms for block in current),
                block_ends=tuple(block.end_ms for block in current),
            )
        )
        if end_index >= len(blocks):
            break
        start_index = max(start_index + 1, end_index - overlap_blocks)
    return chunks


def _build_user_prompt(plan: str, chunk: TranscriptChunk, chunk_index: int, total: int) -> str:
    return f"""PODCAST PLAN
{plan}

TRANSCRIPT CHUNK {chunk_index}/{total}
Allowed timestamp window: {chunk.start_ms} to {chunk.end_ms} milliseconds.

{chunk.text}

Return an object with keep_zones. Use integer millisecond timestamps. A zone must
be narratively useful according to the Podcast Plan and must remain inside the
allowed timestamp window. Use only timestamps printed in the square-bracket
block boundaries. The exact shape is
{{"keep_zones":[{{"start_ms":0,"end_ms":1000,"topic":"Topic"}}]}}.
Return an empty keep_zones array when nothing matches."""


def _schema_for_chunk(chunk: TranscriptChunk) -> dict[str, Any]:
    schema = json.loads(json.dumps(KEEP_ZONES_SCHEMA))
    properties = schema["properties"]["keep_zones"]["items"]["properties"]
    properties["start_ms"]["enum"] = list(chunk.block_starts)
    properties["end_ms"]["enum"] = list(chunk.block_ends)
    return schema


def _validate_keep_zones(
    response: dict[str, Any],
    chunk: TranscriptChunk,
) -> list[KeepZone]:
    raw_zones = response.get("keep_zones")
    if not isinstance(raw_zones, list):
        raise LLMRequestError("LLM JSON must contain a keep_zones array")

    zones: list[KeepZone] = []
    for index, item in enumerate(raw_zones):
        if not isinstance(item, dict):
            raise LLMRequestError(f"keep_zones[{index}] must be an object")
        if set(item) != {"start_ms", "end_ms", "topic"}:
            raise LLMRequestError(
                f"keep_zones[{index}] must contain only start_ms, end_ms, and topic"
            )
        start_ms = item["start_ms"]
        end_ms = item["end_ms"]
        topic = item["topic"]
        if not isinstance(start_ms, int) or isinstance(start_ms, bool):
            raise LLMRequestError(f"keep_zones[{index}].start_ms must be an integer")
        if not isinstance(end_ms, int) or isinstance(end_ms, bool):
            raise LLMRequestError(f"keep_zones[{index}].end_ms must be an integer")
        if not isinstance(topic, str) or not topic.strip():
            raise LLMRequestError(f"keep_zones[{index}].topic must be a non-empty string")
        if start_ms < chunk.start_ms or end_ms > chunk.end_ms or end_ms <= start_ms:
            raise LLMRequestError(
                f"keep_zones[{index}] is outside chunk {chunk.start_ms}-{chunk.end_ms}"
            )
        if start_ms not in chunk.block_starts or end_ms not in chunk.block_ends:
            raise LLMRequestError(
                f"keep_zones[{index}] must use transcript block boundary timestamps"
            )
        zones.append(KeepZone(start_ms=start_ms, end_ms=end_ms, topic=topic.strip()))
    return zones


def merge_keep_zones(
    zones: list[KeepZone],
    adjacent_gap_ms: int = 1_000,
) -> list[KeepZone]:
    if not zones:
        return []

    ordered = sorted(zones, key=lambda zone: (zone.start_ms, zone.end_ms, zone.topic))
    merged: list[KeepZone] = [ordered[0]]
    for zone in ordered[1:]:
        previous = merged[-1]
        same_topic = previous.topic.casefold() == zone.topic.casefold()
        overlaps = zone.start_ms <= previous.end_ms
        adjacent_same_topic = same_topic and zone.start_ms <= previous.end_ms + adjacent_gap_ms
        if overlaps or adjacent_same_topic:
            topics = [previous.topic]
            if not same_topic:
                topics.append(zone.topic)
            merged[-1] = KeepZone(
                start_ms=previous.start_ms,
                end_ms=max(previous.end_ms, zone.end_ms),
                topic=" / ".join(topics),
            )
        else:
            merged.append(zone)
    return merged


class RagNarrativeFilter:
    def __init__(
        self,
        llm_client: JsonLLMClient,
        max_transcript_chars: int = 24_000,
    ):
        self.llm_client = llm_client
        self.max_transcript_chars = max_transcript_chars

    def filter(
        self,
        podcast_plan: str,
        transcript: list[TranscriptWord],
    ) -> list[KeepZone]:
        plan = podcast_plan.strip()
        if not plan:
            raise ValueError("Podcast Plan must not be empty")
        if not transcript:
            return []

        blocks = build_transcript_blocks(transcript)
        chunks = chunk_transcript_blocks(blocks, self.max_transcript_chars)
        candidates: list[KeepZone] = []
        for index, chunk in enumerate(chunks, start=1):
            response = self.llm_client.generate_json(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=_build_user_prompt(plan, chunk, index, len(chunks)),
                json_schema=_schema_for_chunk(chunk),
            )
            candidates.extend(_validate_keep_zones(response, chunk))
        return merge_keep_zones(candidates)


def save_keep_zones_json(zones: list[KeepZone], output_path: str | Path) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"keep_zones": [asdict(zone) for zone in zones]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter a word-level transcript against a Podcast Plan using an LLM."
    )
    parser.add_argument("podcast_plan", help="UTF-8 text file containing the Podcast Plan")
    parser.add_argument("transcript", help="Module 2 word-level transcript JSON")
    parser.add_argument("--output", default="output/keep_zones.json")
    parser.add_argument(
        "--backend",
        choices=("http", "huggingface"),
        default=os.getenv("LLM_BACKEND", "http"),
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions"),
    )
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    parser.add_argument("--api-key-env", default="LLM_API_KEY")
    parser.add_argument("--max-transcript-chars", type=int, default=None)
    parser.add_argument("--hf-model-path", default=os.getenv("HF_MODEL_PATH"))
    parser.add_argument("--hf-repo", default=os.getenv("HF_MODEL_REPO", DEFAULT_HF_REPO))
    parser.add_argument(
        "--hf-filename",
        default=os.getenv("HF_MODEL_FILENAME", DEFAULT_HF_FILENAME),
    )
    parser.add_argument(
        "--hf-model-directory",
        default=os.getenv("HF_MODEL_DIRECTORY", "models/qwen2.5-3b-instruct-gguf"),
    )
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--hf-context-size", type=int, default=8_192)
    parser.add_argument("--hf-gpu-layers", type=int, default=0)
    parser.add_argument("--hf-threads", type=int, default=None)
    parser.add_argument("--hf-max-output-tokens", type=int, default=2_048)
    return parser


def main() -> None:
    load_dotenv()
    args = _build_argument_parser().parse_args()
    plan = load_podcast_plan(args.podcast_plan)
    transcript = load_transcript_words(args.transcript)
    max_transcript_chars = args.max_transcript_chars or (
        12_000 if args.backend == "huggingface" else 24_000
    )

    if args.backend == "huggingface":
        model_path = (
            Path(args.hf_model_path).expanduser().resolve()
            if args.hf_model_path
            else download_gguf_model(
                repo_id=args.hf_repo,
                filename=args.hf_filename,
                model_directory=args.hf_model_directory,
                token=os.getenv(args.hf_token_env),
            )
        )
        client_context = HuggingFaceGGUFJsonClient(
            model_path=model_path,
            context_size=args.hf_context_size,
            gpu_layers=args.hf_gpu_layers,
            threads=args.hf_threads,
            max_output_tokens=args.hf_max_output_tokens,
        )
    else:
        if not args.model:
            raise SystemExit("Missing --model or LLM_MODEL for the HTTP backend")
        client_context = OpenAICompatibleJsonClient(
            endpoint=args.endpoint,
            model=args.model,
            api_key=os.getenv(args.api_key_env),
        )

    with client_context as client:
        narrative_filter = RagNarrativeFilter(
            client,
            max_transcript_chars=max_transcript_chars,
        )
        zones = narrative_filter.filter(plan, transcript)

    output_path = save_keep_zones_json(zones, args.output)
    print(f"Selected {len(zones)} keep zone(s).")
    print(f"Keep zones: {output_path}")


if __name__ == "__main__":
    main()
