from rich.console import Console

from src.media.audio import extract_audio
from src.media.ffmpeg import check_ffmpeg
from src.media.frame import extract_frame
from src.media.metadata import get_video_metadata
from src.ai.speech.transcribe import SpeechTranscriber, save_transcript_json
from src.ai.vision.scene_detect import detect_scenes, save_scenes_json
from src.analysis.merge import merge_scenes_with_transcript
from src.analysis.scoring.highlight import score_scenes
from src.timeline.builder import build_timeline, save_timeline_json
from src.exporter.preview import render_preview
console = Console()


def main():
    video_path = "assets/podcast.mp4"

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

    console.print("[yellow]Transcribing audio...[/yellow]")

    transcriber = SpeechTranscriber(
        model_name="small",
        device="cpu",
        compute_type="int8",
    )

    segments = transcriber.transcribe(str(audio_path))

    transcript_path = save_transcript_json(
        segments=segments,
        output_path="temp/transcript.json",
    )

    console.print(f"[green]Transcript saved:[/green] {transcript_path}")

    for segment in segments[:5]:
        console.print(
            f"[cyan]{segment.start:.2f} -> {segment.end:.2f}[/cyan] {segment.text}"
        )

    console.print("[yellow]Detecting scenes...[/yellow]")

    scenes = detect_scenes(video_path)

    scenes_path = save_scenes_json(
        scenes=scenes,
        output_path="temp/scenes.json",
    )

    console.print(f"[green]Scenes saved:[/green] {scenes_path}")

    for scene in scenes[:5]:
        console.print(
            f"[magenta]{scene.start:.2f} -> {scene.end:.2f}[/magenta]"
        )
    console.print("[yellow]Merging scenes with transcript...[/yellow]")

    analyzed_scenes = merge_scenes_with_transcript(
        scenes=scenes,
        transcript=segments,
    )

    for item in analyzed_scenes[:5]:
        console.print(
            f"[blue]{item.start:.2f} -> {item.end:.2f}[/blue] {item.text}"
        )

    console.print("[yellow]Scoring scenes...[/yellow]")

    scored_scenes = score_scenes(analyzed_scenes)

    for item in scored_scenes[:5]:
        console.print(
            f"[green]score={item.score:.0f}[/green] "
            f"[blue]{item.start:.2f}->{item.end:.2f}[/blue] "
            f"{item.reason} | {item.text[:80]}"
        )

    console.print("[yellow]Building timeline...[/yellow]")

    timeline = build_timeline(
        scored_scenes=scored_scenes,
        min_score=60,
        max_clips=10,
    )

    timeline_path = save_timeline_json(
        timeline=timeline,
        output_path="output/timeline.json",
    )

    console.print(f"[green]Timeline saved:[/green] {timeline_path}")

    for clip in timeline:
        console.print(
            f"[cyan]{clip.timeline_start:.2f}->{clip.timeline_end:.2f}[/cyan] "
            f"from "
            f"[blue]{clip.source_start:.2f}->{clip.source_end:.2f}[/blue] "
            f"score={clip.score:.0f}"
        )
    console.print("[yellow]Rendering preview video...[/yellow]")

    preview_path = render_preview(
        input_video=video_path,
        timeline=timeline,
        output_video="output/preview.mp4",
    )

    console.print(f"[green]Preview video saved:[/green] {preview_path}")

if __name__ == "__main__":
    main()