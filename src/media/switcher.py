from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from src.analysis.schemas import CameraDecision, KeepZone


def _read_wave(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        dtype = np.int16
    elif sampwidth == 4:
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    data = np.frombuffer(raw, dtype=dtype).astype(np.float64)

    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)

    max_val = float(np.iinfo(dtype).max)
    data /= max_val

    return data, sample_rate


def _highpass_filter(data: np.ndarray, cutoff_hz: float, sample_rate: int) -> np.ndarray:
    window = int(sample_rate / cutoff_hz)
    if window < 2:
        return data
    kernel = np.ones(window) / window
    lowpass = np.convolve(data, kernel, mode="same")
    return data - lowpass


def _rms_dbfs(chunk: np.ndarray) -> float:
    if len(chunk) == 0:
        return -np.inf
    rms = np.sqrt(np.mean(chunk ** 2))
    if rms <= 0:
        return -np.inf
    return float(20.0 * np.log10(rms))


def compute_camera_decisions(
    track_a_path: str | Path,
    track_b_path: str | Path,
    keep_zones: list[KeepZone],
    sample_rate: int = 16000,
    highpass_hz: float = 90.0,
    window_ms: int = 100,
    hold_ms: int = 1500,
    silence_threshold_dbfs: float = -45.0,
    active_threshold_db: float = 4.0,
    crosstalk_threshold_db: float = 2.0,
) -> list[CameraDecision]:
    raw_a, sr_a = _read_wave(str(track_a_path))
    raw_b, sr_b = _read_wave(str(track_b_path))

    if sr_a != sr_b:
        raise ValueError(f"Sample rate mismatch: {sr_a} vs {sr_b}")
    if sr_a != sample_rate:
        raise ValueError(f"Expected {sample_rate} Hz, got {sr_a} Hz")

    filtered_a = _highpass_filter(raw_a, highpass_hz, sr_a)
    filtered_b = _highpass_filter(raw_b, highpass_hz, sr_b)

    window_samples = int(sr_a * window_ms / 1000)
    decisions: list[CameraDecision] = []
    current_camera: str | None = None
    segment_start_ms: int | None = None
    hold_until_ms: int = 0

    for zone in keep_zones:
        zone_start_sample = int(zone.start_ms * sr_a / 1000)
        zone_end_sample = int(zone.end_ms * sr_a / 1000)

        for win_start_sample in range(zone_start_sample, zone_end_sample, window_samples):
            win_end_sample = min(win_start_sample + window_samples, zone_end_sample)
            win_start_ms = int(win_start_sample * 1000 / sr_a)
            win_end_ms = int(win_end_sample * 1000 / sr_a)

            chunk_a = filtered_a[win_start_sample:win_end_sample]
            chunk_b = filtered_b[win_start_sample:win_end_sample]

            dbfs_a = _rms_dbfs(chunk_a)
            dbfs_b = _rms_dbfs(chunk_b)

            # Raw camera decision for this window
            if dbfs_a > silence_threshold_dbfs and dbfs_b > silence_threshold_dbfs:
                diff = dbfs_a - dbfs_b
                if abs(diff) < crosstalk_threshold_db:
                    raw_cam = "cam_wide"
                elif diff >= active_threshold_db:
                    raw_cam = "cam_a"
                elif diff <= -active_threshold_db:
                    raw_cam = "cam_b"
                else:
                    raw_cam = "cam_wide"
            elif dbfs_a > silence_threshold_dbfs:
                raw_cam = "cam_a"
            elif dbfs_b > silence_threshold_dbfs:
                raw_cam = "cam_b"
            else:
                raw_cam = "silent"

            if raw_cam == "silent":
                if current_camera is not None:
                    decisions.append(CameraDecision(
                        start_ms=segment_start_ms,
                        end_ms=win_start_ms,
                        camera=current_camera,
                    ))
                    current_camera = None
                    segment_start_ms = None
                continue

            if current_camera is None:
                current_camera = raw_cam
                segment_start_ms = win_start_ms
                hold_until_ms = win_end_ms + hold_ms
            elif win_start_ms < hold_until_ms:
                continue
            elif raw_cam != current_camera:
                decisions.append(CameraDecision(
                    start_ms=segment_start_ms,
                    end_ms=win_start_ms,
                    camera=current_camera,
                ))
                current_camera = raw_cam
                segment_start_ms = win_start_ms
                hold_until_ms = win_end_ms + hold_ms

    if current_camera is not None and segment_start_ms is not None:
        last_zone_end = keep_zones[-1].end_ms if keep_zones else 0
        decisions.append(CameraDecision(
            start_ms=segment_start_ms,
            end_ms=last_zone_end,
            camera=current_camera,
        ))

    return decisions


def save_camera_decisions(
    decisions: list[CameraDecision],
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [
        {"start_ms": d.start_ms, "end_ms": d.end_ms, "camera": d.camera}
        for d in decisions
    ]
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
