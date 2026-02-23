## dinscribe audio transcription
Processes audio through a three-step pipeline to produce a transcription JSON: denoise, voice activity detection, transcribe.

### Setup
```bash
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
```


### Run the full pipeline
```bash
python main.py input/audio.mp3          # Run for a single file
python main.py input/                   # Run for all audio files in a folder
python main.py input/audio.mp3 -f       # Force re-run all steps
```
Each step checks whether its output already exists and skips it if so. Use `-f` to force all steps to re-run regardless, `-o <output_dir>` to specify a different output directory, and `-c <config.yaml>` to specify a different config file.

Output is written to `output/<filename>/` and contains:
- `<filename>_denoised.wav` (vocals isolated from background noise)
- `<filename>_vad.json` (detected speech segment boundaries)
- `<filename>_transcription.json` (final transcription with timestamps)


### Configuration
Edit `config.yaml` to adjust settings for each step. Some important options are:
- `denoise.model` - Demucs model for vocal isolation (default: `htdemucs`)
- `vad.threshold` - VAD speech detection sensitivity (default: `0.5`)
- `transcribe.model` - Whisper model size `tiny` through `large` (default: `base`)
- `transcribe.language` - Transcription language code (default: `en`)


#### Other tips for best results
Add domain-specific vocabulary to `vocab.txt` to improve transcription accuracy on unusual words and jargon. For noisy or technical audio, set `temperature: 0` to disable attempts to fallback to higher-temperature decoding, and consider filtering out any common hallucinations specific to your dataset.


### Run individual steps
Each step can also be run alone:
```bash
python denoise.py audio.mp3
python vad.py audio_denoised.wav
python transcribe.py audio_denoised.wav audio_vad.json
```
