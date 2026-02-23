"""
Uses Demucs to isolate vocals in audio.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from utils import AUDIO_EXTENSIONS, load_config, cuda_available, warn_if_no_cuda


def _run_demucs(input_file: Path, output_file: Path, model: str = "htdemucs", device: str = "cpu"):
    """Run Demucs and copy the vocals stem to the destination path."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        result = subprocess.run(
            [sys.executable, "-m", "demucs", "-n", model, "--two-stems=vocals",
             "--device", device, "-o", str(tmp_path), str(input_file)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"demucs exited with code {result.returncode}")

        vocals_path = tmp_path / model / input_file.stem / "vocals.wav"
        if not vocals_path.exists():
            raise FileNotFoundError(f"Demucs output not found: {vocals_path}")

        shutil.copy2(vocals_path, output_file)


def run(input_path: Path, output_dir: Path, config: dict, force: bool = False) -> Path:
    """Denoise a single audio file using Demucs vocal isolation. Returns path to denoised .wav."""
    output_file = output_dir / f"{output_dir.name}_denoised.wav"
    if not force and output_file.exists():
        return output_file

    model = config.get("model", "htdemucs")
    device = "cuda" if cuda_available() else "cpu"
    output_dir.mkdir(parents=True, exist_ok=True)
    _run_demucs(input_path, output_file, model=model, device=device)
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="Denoise audio using Demucs vocal isolation"
    )
    parser.add_argument("input")
    parser.add_argument("-o", "--output", default="output")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, "denoise")
    warn_if_no_cuda()
    input_path = Path(args.input)
    output_dir = Path(args.output)

    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        files = sorted(f for f in input_path.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS)
        if not files:
            print(f"No audio files found in {input_path}")
            sys.exit(1)
    else:
        print(f"Not found: {input_path}")
        sys.exit(1)

    for audio_file in files:
        out_file = output_dir / audio_file.stem / f"{audio_file.stem}_denoised.wav"
        if not args.force and out_file.exists():
            print(f"Skipping {audio_file.name} (cached)")
            print(f"  -> {out_file}")
            continue
        print(f"Denoising {audio_file.name}...")
        try:
            out = run(audio_file, output_dir / audio_file.stem, config, force=args.force)
            print(f"  -> {out}")
        except Exception as e:
            print(f"  FAILED: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
