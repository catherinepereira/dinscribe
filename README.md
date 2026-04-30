## dinscribe audio transcription
Processes audio through a three-step pipeline to produce a transcription JSON: denoising (demucs), voice activity detection (Silero VAD), and transcription (Whisper).

### Installation
```bash
pip install dinscribe
```

On first run, dinscribe copies default config files to your platform config directory:
- **Windows:** `%APPDATA%\dinscribe\`
- **macOS:** `~/Library/Application Support/dinscribe/`
- **Linux:** `~/.config/dinscribe/`

Edit `config.yaml` and `vocab.txt` to customize settings.


### CLI usage
```bash
dinscribe input/audio.mp3          # single file
dinscribe input/                   # all audio files in a folder
dinscribe input/audio.mp3 -f       # force re-run all steps
dinscribe input/audio.mp3 -c path/to/config.yaml   # custom config
dinscribe input/audio.mp3 -o results/              # custom output dir
```

Each step checks whether its output already exists and skips it if so. Use `-f` to force all steps to re-run.

Output is written to `output/<filename>/` and contains:
- `<filename>_denoised.wav` (vocals isolated from background noise)
- `<filename>_vad.json` (detected speech segment boundaries)
- `<filename>_transcription.json` (final transcription with timestamps)


### Python API
```python
from pathlib import Path
import dinscribe
from dinscribe import PipelineConfig, VadConfig, TranscribeConfig

# Run the full pipeline with defaults
dinscribe.process_file(
    input_path=Path("recording.wav"),
    output_dir=Path("output"),
)

# Custom config
config = PipelineConfig(
    vad=VadConfig(threshold=0.4, max_segment_length_sec=20),
    transcribe=TranscribeConfig(model="small", language="en"),
)
dinscribe.process_file(Path("recording.wav"), Path("output"), config=config)

# Or use individual stages
from dinscribe import denoise, vad, transcribe

denoised = denoise.run(Path("recording.wav"), Path("output/recording"))
vad_file  = vad.run(denoised, Path("output/recording"))
result    = transcribe.run(denoised, vad_file, Path("output/recording"))
```


### Configuration

```yaml
denoise:
  model: htdemucs        # htdemucs | htdemucs_ft | mdx | mdx_extra | htdemucs_6s

vad:
  threshold: 0.5         # 0.0–1.0, higher = requires clearer speech
  min_speech_duration_ms: 250
  min_silence_duration_ms: 100
  padding_ms: 500
  max_segment_length_sec: 30
  merge_within_sec: 1.0

transcribe:
  model: base            # tiny | base | small | medium | large
  language: en           # set to null to auto-detect
  temperature: null      # null = Whisper fallback sequence, 0 = greedy
  no_speech_threshold: 0.6
  logprob_threshold: -1.0
  compression_ratio_threshold: 2.4
  condition_on_previous_text: false
  vocab_file: null       # path to domain-specific vocabulary, defaults to vocab.txt in config dir
```

Add domain-specific vocabulary to `vocab.txt` to improve transcription accuracy on unusual words and jargon. For noisy or technical audio, set `temperature: 0` to disable attempts to fallback to higher-temperature decoding, and consider filtering out any common hallucinations specific to your dataset.
