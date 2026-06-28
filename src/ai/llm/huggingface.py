from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

from huggingface_hub import hf_hub_download


DEFAULT_HF_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
DEFAULT_HF_FILENAME = "qwen2.5-3b-instruct-q4_k_m.gguf"


def download_gguf_model(
    repo_id: str = DEFAULT_HF_REPO,
    filename: str = DEFAULT_HF_FILENAME,
    model_directory: str | Path = "models/qwen2.5-3b-instruct-gguf",
    token: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """Download one GGUF file from Hugging Face into a project-local model folder."""
    target_directory = Path(model_directory).expanduser().resolve()
    target_directory.mkdir(parents=True, exist_ok=True)
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=target_directory,
        token=token,
        local_files_only=local_files_only,
    )
    return Path(downloaded_path).resolve()


def _parse_json_response(content: str) -> dict[str, Any]:
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
        raise RuntimeError("Local Hugging Face model returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Local Hugging Face model JSON root must be an object")
    return value


class HuggingFaceGGUFJsonClient:
    """Run a Hugging Face GGUF instruct model in-process through llama.cpp."""

    def __init__(
        self,
        model_path: str | Path,
        context_size: int = 8_192,
        gpu_layers: int = -1,
        threads: int | None = None,
        max_output_tokens: int = 2_048,
        llama_factory: Callable[..., Any] | None = None,
    ):
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"GGUF model does not exist: {path}")
        if context_size < 2_048:
            raise ValueError("context_size must be at least 2048")
        if max_output_tokens < 128:
            raise ValueError("max_output_tokens must be at least 128")

        if llama_factory is None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError(
                    "llama-cpp-python is required for the Hugging Face GGUF backend."
                ) from exc
            llama_factory = Llama

        resolved_threads = threads or max(1, (os.cpu_count() or 2) // 2)
        self.max_output_tokens = max_output_tokens
        self._llm = llama_factory(
            model_path=str(path),
            n_ctx=context_size,
            n_gpu_layers=gpu_layers,
            n_threads=resolved_threads,
            n_threads_batch=resolved_threads,
            seed=0,
            verbose=False,
        )

    def close(self) -> None:
        close_method = getattr(self._llm, "close", None)
        if callable(close_method):
            close_method()

    def __enter__(self) -> HuggingFaceGGUFJsonClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
    ) -> dict[str, Any]:
        response = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            seed=0,
            max_tokens=self.max_output_tokens,
            response_format={"type": "json_object", "schema": json_schema},
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Local Hugging Face model returned an invalid envelope") from exc
        if not isinstance(content, str):
            raise RuntimeError("Local Hugging Face model returned no JSON content")
        return _parse_json_response(content)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the default local GGUF model.")
    parser.add_argument("--repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--filename", default=DEFAULT_HF_FILENAME)
    parser.add_argument("--model-directory", default="models/qwen2.5-3b-instruct-gguf")
    parser.add_argument("--token-env", default="HF_TOKEN")
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    model_path = download_gguf_model(
        repo_id=args.repo,
        filename=args.filename,
        model_directory=args.model_directory,
        token=os.getenv(args.token_env),
        local_files_only=args.local_files_only,
    )
    print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
