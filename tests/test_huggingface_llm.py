import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.ai.llm.huggingface import (
    HuggingFaceGGUFJsonClient,
    download_gguf_model,
)


class FakeLlama:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.call_kwargs = None
        self.closed = False

    def create_chat_completion(self, **kwargs):
        self.call_kwargs = kwargs
        return {
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
        }

    def close(self):
        self.closed = True


class HuggingFaceGGUFClientTests(unittest.TestCase):
    def test_runs_grammar_constrained_chat_completion(self) -> None:
        instances = []

        def factory(**kwargs):
            instance = FakeLlama(**kwargs)
            instances.append(instance)
            return instance

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            model_path.touch()
            with HuggingFaceGGUFJsonClient(
                model_path,
                context_size=4_096,
                gpu_layers=5,
                threads=2,
                llama_factory=factory,
            ) as client:
                result = client.generate_json(
                    "system",
                    "user",
                    {"type": "object"},
                )

        self.assertIn("keep_zones", result)
        self.assertEqual(instances[0].init_kwargs["n_gpu_layers"], 5)
        self.assertEqual(
            instances[0].call_kwargs["response_format"],
            {"type": "json_object", "schema": {"type": "object"}},
        )
        self.assertTrue(instances[0].closed)

    def test_rejects_missing_model_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            HuggingFaceGGUFJsonClient("missing.gguf", llama_factory=FakeLlama)


class HuggingFaceDownloadTests(unittest.TestCase):
    @patch("src.ai.llm.huggingface.hf_hub_download")
    def test_downloads_selected_gguf_into_local_directory(self, download_mock) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.gguf"
            download_mock.return_value = str(model_path)
            result = download_gguf_model(
                repo_id="org/model",
                filename="model.gguf",
                model_directory=directory,
                token="token",
            )

        self.assertEqual(result, model_path.resolve())
        self.assertEqual(download_mock.call_args.kwargs["repo_id"], "org/model")
        self.assertEqual(download_mock.call_args.kwargs["filename"], "model.gguf")


if __name__ == "__main__":
    unittest.main()
