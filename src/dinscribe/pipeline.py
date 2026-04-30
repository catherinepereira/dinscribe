import json
import time
from pathlib import Path
from . import denoise, vad, transcribe
from .utils import fmt_time, progress_bar, cuda_available
from .config import PipelineConfig


def process_file(input_path: Path, output_dir: Path, config: PipelineConfig = PipelineConfig(), force: bool = False) -> bool:
    """Run the full pipeline for one audio file."""
    file_dir = output_dir / input_path.stem
    file_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.monotonic()

    print(f"\n{'─' * 60}")
    print(f"  Input: {input_path}")
    print(f"  Output: {file_dir}/")
    print('─' * 60)

    print("\n  [1/3] Denoising")
    denoised_path = file_dir / f"{input_path.stem}_denoised.wav"
    if not force and denoised_path.exists():
        print("        ✓ Skipped (cached)")
    else:
        print("        Running Demucs vocal isolation...")
        step_start = time.monotonic()
        try:
            denoised_path = denoise.run(input_path, file_dir, config.denoise, force=force)
        except Exception as e:
            print(f"\n  ERROR: Denoising failed: {e}")
            return False
        print(f"        ✓ Done  ({fmt_time(time.monotonic() - step_start)})")

    print("\n  [2/3] Voice Activity Detection")
    vad_path = file_dir / f"{input_path.stem}_vad.json"
    vad_cached = not force and vad_path.exists()
    step_start = time.monotonic()
    if vad_cached:
        print("        ✓ Skipped (cached)")
    else:
        print("        Detecting speech segments...")
        try:
            vad_path = vad.run(denoised_path, file_dir, config.vad, force=force)
        except Exception as e:
            print(f"\n  ERROR: VAD failed: {e}")
            return False

    vad_meta = json.loads(vad_path.read_text(encoding="utf-8"))["metadata"]
    segment_count = vad_meta["segment_count"]
    if not vad_cached:
        speech_s = vad_meta["total_speech_ms"] / 1000
        audio_s = vad_meta["audio_duration_ms"] / 1000
        print(f"        ✓ {segment_count} segments found  "
              f"({speech_s:.1f}s speech in {audio_s:.1f}s audio)  "
              f"({fmt_time(time.monotonic() - step_start)})")

    print("\n  [3/3] Transcribing")
    transcription_path = file_dir / f"{input_path.stem}_transcription.json"
    trans_cached = not force and transcription_path.exists()
    if trans_cached:
        print("        ✓ Skipped (cached)")
    else:
        device = "cuda" if cuda_available() else "cpu"
        print(f"        Loading Whisper '{config.transcribe.model}' on {device}...")

        step_start = time.monotonic()

        def on_segment(current: int, total: int):
            bar = progress_bar(current, total)
            print(f"\r        [{bar}] {current}/{total} segments", end="", flush=True)

        try:
            transcription_path = transcribe.run(
                denoised_path, vad_path, file_dir, config.transcribe, on_segment=on_segment, force=force,
            )
        except Exception as e:
            print(f"\n  ERROR: Transcription failed: {e}")
            return False

        print()
        trans_meta = json.loads(transcription_path.read_text(encoding="utf-8"))["metadata"]
        kept = trans_meta["processed_segments"]
        print(f"        ✓ {kept}/{segment_count} segments processed  "
              f"({fmt_time(time.monotonic() - step_start)})")

    print(f"\n{'─' * 60}")
    print(f"  Total time: {fmt_time(time.monotonic() - total_start)}")
    print("  Output:")
    for p in (denoised_path, vad_path, transcription_path):
        print(f"    {p.relative_to(output_dir.parent)}")
    print('─' * 60)

    return True
