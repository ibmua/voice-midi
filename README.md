# Voice to MIDI

Convert voice recordings and audio files into MIDI melody files. Sing or hum into your microphone (or upload a file) and get a playable MIDI file out.

## Features

- **Multi-format audio input** -- WAV, MP3, FLAC, OGG, M4A
- **Microphone recording** -- record directly in the browser
- **Ensemble pitch detection** -- fuses Praat autocorrelation, pYIN, and YIN for accurate voice tracking
- **Smart note segmentation** -- onset detection, pitch-jump splitting, and short-note merging
- **Interactive piano roll** -- color-coded notes with confidence-based opacity and a real-time playhead
- **Simultaneous playback** -- play original audio and MIDI preview together, DAW-style
- **Scale quantization** -- snap notes to major, minor, pentatonic, blues, or chromatic scales
- **General MIDI instruments** -- choose from 32+ instruments for export
- **Pitch bend support** -- preserves expressive pitch curves in the MIDI output
- **One-click MIDI export**

## Quick Start

```bash
# Clone
git clone https://github.com/ibmua/voice-midi.git
cd voice-midi

# Set up environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
python3 main.py
```

Opens automatically at [http://localhost:5111](http://localhost:5111).

### Optional: ffmpeg

MP3, M4A, and OGG support requires [ffmpeg](https://ffmpeg.org/). WAV and FLAC work without it.

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg
```

## How It Works

1. **Load audio** -- upload a file or record from your mic
2. **Detect pitch** -- runs a 3-detector ensemble (Praat + pYIN + YIN) with RMS energy gating
3. **Segment notes** -- splits the pitch curve into discrete note events using onsets and pitch jumps
4. **Generate MIDI** -- converts notes to MIDI with tempo, instrument, optional scale filtering, and pitch bends
5. **Preview & export** -- play back original + MIDI simultaneously, then download the .mid file

## Project Structure

```
voice-midi/
  main.py                          # Entry point (Flask server on port 5111)
  requirements.txt
  voice_midi/
    pipeline.py                    # Orchestrates the full conversion workflow
    web/server.py                  # Flask app + embedded web UI
    audio/
      loader.py                    # Multi-format audio loading + resampling
      recorder.py                  # Microphone recording via sounddevice
      player.py                    # Desktop audio playback
    analysis/
      pitch_detector.py            # 3-detector ensemble pitch tracking
      note_segmenter.py            # Onset detection + note extraction
      music_utils.py               # MIDI/frequency helpers, scales, instruments
    midi/
      generator.py                 # MIDI file generation with pitch bends
      synthesizer.py               # Sine wave synthesis for preview
```

## Requirements

- Python 3.9+
- See [requirements.txt](requirements.txt) for dependencies

## License

[MIT](LICENSE)
