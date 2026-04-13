#!/usr/bin/env python3
"""Voice to MIDI — Convert voice recordings into MIDI melody files."""

import os
import sys
import shutil
import webbrowser
import threading

os.environ['TK_SILENCE_DEPRECATION'] = '1'


def check_dependencies():
    missing = []
    for name in ('flask', 'librosa', 'soundfile', 'sounddevice', 'midiutil', 'numpy'):
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print("Install: pip install -r requirements.txt")
        sys.exit(1)
    if not shutil.which("ffmpeg"):
        print("Note: ffmpeg not found — MP3/M4A/OGG need it. WAV/FLAC work without it.")


if __name__ == "__main__":
    check_dependencies()
    from voice_midi.web.server import app

    port = 5111
    print(f"\n  Voice to MIDI — http://localhost:{port}\n")
    threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{port}')).start()
    app.run(host='127.0.0.1', port=port, debug=False)
