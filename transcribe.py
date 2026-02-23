"""
Transcribes speech segments from denoised audio using Whisper.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional
from pydub import AudioSegment
import whisper
from utils import load_config, progress_bar, cuda_available, warn_if_no_cuda

# Loaded once per process, keyed by (model_name, device)
_whisper_cache: dict = {}


def _load_model(model_name: str, device: str):
    key = (model_name, device)
    if key not in _whisper_cache:
        _whisper_cache[key] = whisper.load_model(model_name, device=device)
    return _whisper_cache[key]


def _load_vocabulary(vocab_file: Optional[str]) -> str:
    """Build Whisper's initial_prompt from a vocabulary file."""
    if not vocab_file:
        return ""
    vocab_path = Path(vocab_file)
    if not vocab_path.exists():
        return ""
    terms = [
        line.strip()
        for line in vocab_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return f"Common terms: {', '.join(terms)}." if terms else ""



def run(
    audio_path: Path,
    vad_path: Path,
    output_dir: Path,
    config: dict,
    on_segment=None,
    force: bool = False,
) -> Path:
    """Transcribe speech segments from denoised audio. Returns path to transcription.json."""
    output_file = output_dir / f"{output_dir.name}_transcription.json"
    if not force and output_file.exists():
        return output_file

    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = config.get("model", "base")
    language = config.get("language", "en")
    temperature = config.get("temperature", None)
    no_speech_threshold = config.get("no_speech_threshold", None)
    logprob_threshold = config.get("logprob_threshold", None)
    compression_ratio_threshold = config.get("compression_ratio_threshold", None)
    condition_on_previous_text = config.get("condition_on_previous_text", False)
    vocab_file = config.get("vocab_file", "vocab.txt")

    device = "cuda" if cuda_available() else "cpu"
    model = _load_model(model_name, device)
    initial_prompt = _load_vocabulary(vocab_file) or None

    vad_data = json.loads(vad_path.read_text(encoding="utf-8"))
    segments = vad_data.get("segments", [])
    if not segments:
        raise RuntimeError(f"No segments found in {vad_path.name}")

    audio = AudioSegment.from_file(str(audio_path))
    audio_duration_ms = len(audio)

    valid_segments = [
        (seg["start_ms"], min(seg["end_ms"], audio_duration_ms))
        for seg in segments
        if seg["start_ms"] < audio_duration_ms
    ]
    total = len(valid_segments)

    output = {
        "metadata": {
            "source_audio": audio_path.name,
            "model": model_name,
            "language": language,
            "temperature": temperature,
            "no_speech_threshold": no_speech_threshold,
            "logprob_threshold": logprob_threshold,
            "compression_ratio_threshold": compression_ratio_threshold,
            "total_segments": total,
            "processed_segments": 0,
        },
        "transcription": [],
    }
    _write_json(output_file, output)

    temp_dir = output_dir / "_temp_segments"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    try:
        for i, (start_ms, end_ms) in enumerate(valid_segments, 1):
            temp_file = temp_dir / f"seg_{i}.wav"
            audio[start_ms:end_ms].export(str(temp_file), format="wav")

            try:
                transcribe_kwargs = dict(
                    language=language,
                    fp16=(device == "cuda"),
                    condition_on_previous_text=condition_on_previous_text,
                    initial_prompt=initial_prompt,
                )
                if temperature is not None:
                    transcribe_kwargs["temperature"] = temperature
                if no_speech_threshold is not None:
                    transcribe_kwargs["no_speech_threshold"] = no_speech_threshold
                if logprob_threshold is not None:
                    transcribe_kwargs["logprob_threshold"] = logprob_threshold
                if compression_ratio_threshold is not None:
                    transcribe_kwargs["compression_ratio_threshold"] = compression_ratio_threshold
                result = model.transcribe(str(temp_file), **transcribe_kwargs)
            except Exception:
                temp_file.unlink(missing_ok=True)
                if on_segment:
                    on_segment(i, total)
                continue

            temp_file.unlink(missing_ok=True)

            text = result["text"].strip()
            no_speech_prob = max(
                (s.get("no_speech_prob", 0) for s in result.get("segments", [])),
                default=0,
            )

            output["metadata"]["processed_segments"] += 1

            if not text or (no_speech_threshold is not None and no_speech_prob > no_speech_threshold):
                _write_json(output_file, output)
                if on_segment:
                    on_segment(i, total)
                continue

            output["transcription"].append({
                "timestamp": {"start": start_ms / 1000, "end": end_ms / 1000},
                "text": text,
            })
            _write_json(output_file, output)

            if on_segment:
                on_segment(i, total)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    return output_file


def _write_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio segments using Whisper"
    )
    parser.add_argument("audio")
    parser.add_argument("vad")
    parser.add_argument("-o", "--output", default="output")
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("-f", "--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, "transcribe")
    warn_if_no_cuda()
    audio_path = Path(args.audio)
    vad_path = Path(args.vad)
    output_dir = Path(args.output) / audio_path.stem

    if not audio_path.exists():
        print(f"Audio file not found: {audio_path}")
        sys.exit(1)
    if not vad_path.exists():
        print(f"VAD file not found: {vad_path}")
        sys.exit(1)

    def on_segment(current, total):
        bar = progress_bar(current, total, width=30)
        print(f"\r  [{bar}] {current}/{total}", end="", flush=True)
        if current == total:
            print()

    out_file = output_dir / f"{audio_path.stem}_transcription.json"
    if not args.force and out_file.exists():
        print(f"Skipping {audio_path.name} (cached)")
        print(f"  -> {out_file}")
        return

    print(f"Transcribing {audio_path.name}...")
    try:
        out = run(audio_path, vad_path, output_dir, config, on_segment=on_segment, force=args.force)
        meta = json.loads(out.read_text(encoding="utf-8"))["metadata"]
        print(f"  {meta['processed_segments']}/{meta['total_segments']} segments processed")
        print(f"  -> {out}")
    except Exception as e:
        print(f"\n  FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
