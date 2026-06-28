from pathlib import Path

from rich.console import Console

from src.media.audio import extract_audio
from src.media.ffmpeg import check_ffmpeg
from src.media.frame import extract_frame
from src.media.metadata import get_video_metadata
from src.ai.speech.transcribe import SpeechTranscriber, save_word_transcript_json
from src.ai.vision.scene_detect import detect_scenes, save_scenes_json
from src.analysis.merge import merge_scenes_with_transcript
from src.analysis.scoring.highlight import score_scenes, ScoredScene
from src.analysis.rag_filter import (
    RagNarrativeFilter,
    load_podcast_plan,
    save_keep_zones_json,
)
from src.ai.llm.huggingface import HuggingFaceGGUFJsonClient, download_gguf_model
from src.media.switcher import compute_camera_decisions, save_camera_decisions
from src.timeline.builder import (
    build_timeline,
    build_timeline_from_decisions,
    build_timeline_from_zones,
    save_timeline_json,
)
from src.exporter.fcpxml import export_fcpxml
from src.exporter.preview import render_preview
from src.analysis.schemas import KeepZone

console = Console()


def _run_rag_filter(words, plan, model_path):
    """RAG filter using local HuggingFace model (Module 3)."""
    with HuggingFaceGGUFJsonClient(
        model_path=model_path,
        context_size=8_192,
        max_output_tokens=2_048,
    ) as llm_client:
        narrative_filter = RagNarrativeFilter(
            llm_client,
            max_transcript_chars=12_000,
        )
        return narrative_filter.filter(plan, words)


def _run_keyword_scoring(video_path, transcriber, audio_path):
    """Fallback: segment-level scoring without LLM."""
    from src.ai.speech.transcribe import save_transcript_json
    segments = transcriber.transcribe(str(audio_path))
    save_transcript_json(segments, "temp/transcript.json")

    scenes = detect_scenes(video_path)
    save_scenes_json(scenes, "temp/scenes.json")

    analyzed = merge_scenes_with_transcript(scenes, segments)
    scored = score_scenes(analyzed)
    timeline = build_timeline(scored, min_score=60, max_clips=10)
    return timeline, scored


def main():
    video_path = "assets/podcast.mp4"
    podcast_plan_path = "input/podcast_plan_qa.txt"
    model_path = "models/qwen2.5-3b-instruct-gguf/qwen2.5-3b-instruct-q4_k_m.gguf"

    if not check_ffmpeg():
        console.print("[red]FFmpeg chưa cài đúng.[/red]")
        return
    console.print("[green]FFmpeg OK[/green]")

    metadata = get_video_metadata(video_path)
    console.print(metadata)

    audio_path = extract_audio(
        input_video=video_path,
        output_audio="temp/audio.wav",
    )
    console.print(f"[green]Audio exported:[/green] {audio_path}")

    frame_path = extract_frame(
        input_video=video_path,
        timestamp=3.0,
        output_image="output/frame.jpg",
    )
    console.print(f"[green]Frame exported:[/green] {frame_path}")

    from ctranslate2 import get_cuda_device_count
    device = "cuda" if get_cuda_device_count() > 0 else "cpu"
    transcriber = SpeechTranscriber(
        model_name="small",
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
    )

    console.print("[yellow]Transcribing audio (word-level)...[/yellow]")
    words = transcriber.transcribe_words(audio_path, speaker_id="Speaker")
    console.print(f"[green]Transcribed {len(words)} words[/green]")

    transcript_path = save_word_transcript_json(
        words=words,
        output_path="temp/transcript_words.json",
    )
    console.print(f"[green]Word transcript saved:[/green] {transcript_path}")

    plan = load_podcast_plan(podcast_plan_path)
    console.print(f"[green]Podcast Plan:[/green] {plan[:100]}...")

    # --- Module 3: RAG Narrative Filter ---
    console.print("[yellow]Running RAG Narrative Filter...[/yellow]")

    try:
        keep_zones = _run_rag_filter(words, plan, model_path)
        console.print(f"[green]RAG filter selected {len(keep_zones)} keep zone(s)[/green]")
        for zone in keep_zones:
            dur = (zone.end_ms - zone.start_ms) / 1000.0
            console.print(f"  [cyan]{zone.start_ms}ms -> {zone.end_ms}ms[/cyan] ({dur:.1f}s) [yellow]{zone.topic}[/yellow]")

        zones_path = save_keep_zones_json(keep_zones, "output/keep_zones.json")
        console.print(f"[green]Keep zones saved:[/green] {zones_path}")

        # --- Module 4: Audio-Level Diarization & Switching ---
        console.print("[yellow]Running Module 4: Audio diarization & silence removal...[/yellow]")
        camera_decisions = compute_camera_decisions(
            track_a_path=audio_path,
            track_b_path=audio_path,
            keep_zones=keep_zones,
        )
        decisions_path = save_camera_decisions(camera_decisions, "output/camera_decisions.json")
        console.print(f"[green]Camera decisions: {len(camera_decisions)} segment(s)[/green]")

        topic_map = {(z.start_ms, z.end_ms): z.topic for z in keep_zones}
        timeline = build_timeline_from_decisions(camera_decisions, topic_map=topic_map)

    except Exception as exc:
        console.print(f"[yellow]RAG filter failed ({exc}). Falling back to keyword scoring...[/yellow]")
        timeline, _ = _run_keyword_scoring(video_path, transcriber, audio_path)

    # --- Build timeline & render ---
    timeline_path = save_timeline_json(
        timeline=timeline,
        output_path="output/timeline.json",
    )
    console.print(f"[green]Timeline saved:[/green] {timeline_path}")

    for clip in timeline:
        cam_tag = f" [{clip.camera}]" if clip.camera else ""
        console.print(
            f"  [cyan]{clip.timeline_start:.1f}->{clip.timeline_end:.1f}[/cyan] "
            f"from [blue]{clip.source_start:.1f}->{clip.source_end:.1f}[/blue] "
            f"score={clip.score:.0f}{cam_tag} text={clip.text[:60]}"
        )

    if timeline:
        console.print("[yellow]Rendering preview video...[/yellow]")
        preview_path = render_preview(
            input_video=video_path,
            timeline=timeline,
            output_video="output/preview.mp4",
        )
        console.print(f"[green]Preview video saved:[/green] {preview_path}")

        # --- Module 5: FCPXML Exporter ---
        console.print("[yellow]Running Module 5: FCPXML export...[/yellow]")
        fcpxml_path = export_fcpxml(
            timeline=timeline,
            video_path=video_path,
            metadata=metadata,
            output_path="output/timeline.fcpxml",
        )
        console.print(f"[green]FCPXML saved:[/green] {fcpxml_path}")
    else:
        console.print("[red]Timeline rỗng — không có gì để render.[/red]")

    console.print("[bold green]Pipeline complete![/bold green]")


if __name__ == "__main__":
    main()
