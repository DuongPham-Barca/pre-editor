import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.media.ingest import (
    AudioStreamMetadata,
    MediaProbe,
    TimecodeMetadata,
    VideoStreamMetadata,
    _extract_timecode,
    ingest_folder,
    probe_media,
    scan_input_folder,
)


class ScanInputFolderTests(unittest.TestCase):
    def test_scans_supported_extensions_case_insensitively(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "camera_b.MOV").touch()
            (root / "camera_a.mp4").touch()
            (root / "mic_a.WAV").touch()
            (root / "notes.txt").touch()
            nested = root / "nested"
            nested.mkdir()
            (nested / "mic_b.wav").touch()

            direct = scan_input_folder(root)
            recursive = scan_input_folder(root, recursive=True)

            self.assertEqual(
                [path.name for path in direct],
                ["camera_a.mp4", "camera_b.MOV", "mic_a.WAV"],
            )
            self.assertEqual(len(recursive), 4)


class TimecodeTests(unittest.TestCase):
    def test_prefers_tmcd_track_and_detects_drop_frame(self) -> None:
        streams = [
            {
                "index": 0,
                "codec_type": "video",
                "tags": {"timecode": "01:00:00:00"},
            },
            {
                "index": 2,
                "codec_type": "data",
                "codec_tag_string": "tmcd",
                "tags": {"handler_name": "TimeCode", "timecode": "10:00:00;00"},
            },
        ]

        result = _extract_timecode(streams, {"timecode": "02:00:00:00"})

        self.assertEqual(result.value, "10:00:00;00")
        self.assertEqual(result.source, "stream:2:data")
        self.assertTrue(result.drop_frame)
        self.assertEqual(len(result.candidates), 3)

    def test_preserves_bwf_time_reference(self) -> None:
        result = _extract_timecode([], {"time_reference": "96000"})

        self.assertIsNone(result.value)
        self.assertEqual(result.bwf_time_reference_samples, 96000)


class ProbeMediaTests(unittest.TestCase):
    @patch("src.media.ingest.run_command")
    def test_probe_preserves_rational_frame_rate_and_time_base(self, run_command) -> None:
        run_command.return_value = json.dumps(
            {
                "streams": [
                    {
                        "index": 0,
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 1920,
                        "height": 1080,
                        "r_frame_rate": "30000/1001",
                        "avg_frame_rate": "30000/1001",
                        "time_base": "1/30000",
                        "start_time": "0.000000",
                        "tags": {"timecode": "01:00:00;00"},
                    },
                    {
                        "index": 1,
                        "codec_type": "audio",
                        "codec_name": "aac",
                        "sample_rate": "48000",
                        "channels": 2,
                        "channel_layout": "stereo",
                        "time_base": "1/48000",
                    },
                ],
                "format": {
                    "format_name": "mov,mp4",
                    "duration": "120.5",
                    "start_time": "0.0",
                    "tags": {},
                },
            }
        )

        result = probe_media("camera.mov")

        self.assertEqual(result.video.frame_rate, "30000/1001")
        self.assertEqual(result.video.time_base, "1/30000")
        self.assertEqual(result.audio.sample_rate, 48000)
        self.assertEqual(result.timecode.value, "01:00:00;00")


class IngestFolderTests(unittest.TestCase):
    @patch("src.media.ingest.check_ffmpeg", return_value=True)
    @patch("src.media.ingest.extract_audio")
    @patch("src.media.ingest.probe_media")
    def test_creates_manifest_and_unique_normalized_audio_paths(
        self,
        probe_media_mock,
        extract_audio_mock,
        _check_ffmpeg_mock,
    ) -> None:
        probe_media_mock.return_value = MediaProbe(
            format_name="mov,mp4",
            duration_seconds=10.0,
            start_time_seconds=0.0,
            video=VideoStreamMetadata(
                index=0,
                codec_name="h264",
                width=1920,
                height=1080,
                frame_rate="25/1",
                average_frame_rate="25/1",
                time_base="1/12800",
                start_time_seconds=0.0,
            ),
            audio=AudioStreamMetadata(
                index=1,
                codec_name="aac",
                sample_rate=48000,
                channels=2,
                channel_layout="stereo",
                time_base="1/48000",
                start_time_seconds=0.0,
            ),
            timecode=TimecodeMetadata(
                value="01:00:00:00",
                source="stream:0:video",
                drop_frame=False,
                candidates=[],
                bwf_time_reference_samples=None,
            ),
            format_tags={},
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / "input"
            output_root = root / "output"
            input_root.mkdir()
            (input_root / "camera.mov").write_bytes(b"video")
            (input_root / "camera.wav").write_bytes(b"audio")

            manifest, manifest_path = ingest_folder(input_root, output_root)

            self.assertEqual(len(manifest.assets), 2)
            self.assertTrue(manifest_path.exists())
            normalized_paths = [asset.normalized_audio_path for asset in manifest.assets]
            self.assertEqual(len(set(normalized_paths)), 2)
            self.assertTrue(normalized_paths[0].endswith("camera.mov.wav"))
            self.assertTrue(normalized_paths[1].endswith("camera.wav.wav"))
            self.assertEqual(extract_audio_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
