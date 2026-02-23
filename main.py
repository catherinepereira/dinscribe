"""
Chains full audio pre-processing pipeline with transcription
"""

import argparse
import json
import sys
import time
from pathlib import Path
import denoise
import vad
import transcribe
from utils import AUDIO_EXTENSIONS, load_config, progress_bar, fmt_time, cuda_available, warn_if_no_cuda


def _process_file(input_path: Path, output_dir: Path, config: dict, force: bool = False) -> bool:
    """Run the full pipeline for one audio file"""
    file_dir = output_dir / input_path.stem
    file_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.monotonic()

    print(f"\n{'─' * 60}")
    print(f"  Input: {input_path}")
    print(f"  Output: {file_dir}/")
    print('─' * 60)

    # Step 1: Denoise
    print("\n  [1/3] Denoising")
    denoised_path = file_dir / f"{input_path.stem}_denoised.wav"
    if not force and denoised_path.exists():
        print("        ✓ Skipped (cached)")
    else:
        print("        Running Demucs vocal isolation...")
        step_start = time.monotonic()
        try:
            denoised_path = denoise.run(input_path, file_dir, config.get("denoise", {}), force=force)
        except Exception as e:
            print(f"\n  ERROR: Denoising failed: {e}")
            return False
        print(f"        ✓ Done  ({fmt_time(time.monotonic() - step_start)})")

    # Step 2: VAD
    print("\n  [2/3] Voice Activity Detection")
    vad_path = file_dir / f"{input_path.stem}_vad.json"
    vad_cached = not force and vad_path.exists()
    step_start = time.monotonic()
    if vad_cached:
        print("        ✓ Skipped (cached)")
    else:
        print("        Detecting speech segments...")
        try:
            vad_path = vad.run(denoised_path, file_dir, config.get("vad", {}), force=force)
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

    # Step 3: Transcribe
    print("\n  [3/3] Transcribing")
    transcription_path = file_dir / f"{input_path.stem}_transcription.json"
    trans_cached = not force and transcription_path.exists()
    if trans_cached:
        print("        ✓ Skipped (cached)")
    else:
        trans_config = config.get("transcribe", {})
        model_name = trans_config.get("model", "base")
        device = "cuda" if cuda_available() else "cpu"
        print(f"        Loading Whisper '{model_name}' on {device}...")

        step_start = time.monotonic()

        def on_segment(current: int, total: int):
            bar = progress_bar(current, total)
            print(f"\r        [{bar}] {current}/{total} segments", end="", flush=True)

        try:
            transcription_path = transcribe.run(
                denoised_path, vad_path, file_dir, trans_config, on_segment=on_segment, force=force,
            )
        except Exception as e:
            print(f"\n  ERROR: Transcription failed: {e}")
            return False

        print()
        trans_meta = json.loads(transcription_path.read_text(encoding="utf-8"))["metadata"]
        kept = trans_meta["processed_segments"]
        print(f"        ✓ {kept}/{segment_count} segments processed  "
              f"({fmt_time(time.monotonic() - step_start)})")

    # Step 4: Print summary
    print(f"\n{'─' * 60}")
    print(f"  Total time: {fmt_time(time.monotonic() - total_start)}")
    print("  Output:")
    for p in (denoised_path, vad_path, transcription_path):
        print(f"    {p.relative_to(output_dir.parent)}")
    print('─' * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Process audio files into transcriptions"
    )
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default="output")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.is_file():
        if input_path.suffix.lower() not in AUDIO_EXTENSIONS:
            print(f"Unrecognised audio extension: {input_path.suffix}")
            sys.exit(1)
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not files:
            print(f"No audio files found in {input_path}/")
            sys.exit(1)
    else:
        print(f"Not found: {input_path}")
        sys.exit(1)

    print(f"Files to process: {len(files)}")
    print(f"Output directory: {output_dir}/")
    warn_if_no_cuda()

    success = 0
    batch_start = time.monotonic()

    for i, audio_file in enumerate(files, 1):
        if len(files) > 1:
            print(f"\n{'═' * 60}")
            print(f"  File {i}/{len(files)}: {audio_file.name}")

        if _process_file(audio_file, output_dir, config, force=args.force):
            success += 1

    failed = len(files) - success

    if len(files) > 1:
        print(f"\n{'─' * 60}")
        print(f"  Batch complete: {success} succeeded, {failed} failed")
        print(f"  Total time: {fmt_time(time.monotonic() - batch_start)}")
        print(f"\n{'─' * 60}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
