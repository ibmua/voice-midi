import os
import io
import json
import tempfile
import threading
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template_string

from ..pipeline import ConversionPipeline
from ..audio.recorder import MicRecorder
from ..analysis.music_utils import midi_to_note_name, GM_INSTRUMENTS, SCALE_INTERVALS, NOTE_NAMES

app = Flask(__name__)
pipe = ConversionPipeline()
recorder = MicRecorder(sample_rate=22050)

# State
state = {
    'file_name': None,
    'duration': 0,
    'notes': [],
    'midi_ready': False,
    'status': 'Ready',
    'recording': False,
}


def _notes_to_json(notes):
    return [{
        'name': midi_to_note_name(n.midi_note),
        'midi': n.midi_note,
        'start': round(n.start_time, 3),
        'end': round(n.end_time, 3),
        'dur': round(n.duration, 3),
        'vel': n.velocity,
        'conf': round(n.confidence, 2),
    } for n in notes]


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE,
        instruments=json.dumps(list(GM_INSTRUMENTS.keys())[:32]),
        scales=json.dumps(list(SCALE_INTERVALS.keys())),
        notes=json.dumps(NOTE_NAMES),
    )


@app.route('/api/state')
def get_state():
    return jsonify(state)


@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    f = request.files['file']
    tmp = tempfile.NamedTemporaryFile(suffix=os.path.splitext(f.filename)[1], delete=False)
    f.save(tmp.name)
    try:
        pipe.load(tmp.name)
        dur = len(pipe.audio_data) / pipe.sample_rate
        state.update(file_name=f.filename, duration=round(dur, 1), notes=[], midi_ready=False,
                     status=f'Loaded: {f.filename} ({dur:.1f}s)')
        return jsonify({'ok': True, 'duration': dur, 'name': f.filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    finally:
        os.unlink(tmp.name)


@app.route('/api/record/start', methods=['POST'])
def rec_start():
    try:
        recorder.start()
        state['recording'] = True
        state['status'] = 'Recording...'
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/record/stop', methods=['POST'])
def rec_stop():
    audio = recorder.stop()
    state['recording'] = False
    if audio is None or len(audio) < 1000:
        state['status'] = 'Recording too short'
        return jsonify({'error': 'Too short'}), 400

    pipe.audio_data = audio
    pipe.sample_rate = recorder.sample_rate
    pipe.pitch_result = None
    pipe.notes = None
    pipe.midi_file = None

    dur = len(audio) / recorder.sample_rate
    state.update(file_name=f'Recording ({dur:.1f}s)', duration=round(dur, 1),
                 notes=[], midi_ready=False, status=f'Recorded {dur:.1f}s')
    return jsonify({'ok': True, 'duration': dur})


@app.route('/api/detect', methods=['POST'])
def detect():
    if pipe.audio_data is None:
        return jsonify({'error': 'No audio loaded'}), 400

    opts = request.json or {}
    state['status'] = 'Detecting pitch...'
    try:
        pipe.run_pitch_detection(
            fmin=opts.get('fmin', 65.0),
            fmax=opts.get('fmax', 2093.0),
        )
        pipe.run_note_segmentation(
            min_note_duration=opts.get('min_dur', 0.08),
            voiced_threshold=opts.get('voiced_thresh', 0.5),
        )
        notes_json = _notes_to_json(pipe.notes)
        state['notes'] = notes_json
        state['status'] = f'Detected {len(pipe.notes)} notes'
        return jsonify({'ok': True, 'notes': notes_json, 'count': len(pipe.notes)})
    except Exception as e:
        state['status'] = f'Error: {e}'
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate', methods=['POST'])
def generate():
    if not pipe.notes:
        return jsonify({'error': 'No notes detected'}), 400

    opts = request.json or {}
    sf = None
    if opts.get('scale', 'chromatic') != 'chromatic':
        sf = (opts.get('root', 'C'), opts['scale'])

    try:
        pipe.run_midi_generation(
            tempo=opts.get('tempo', 120),
            instrument=GM_INSTRUMENTS.get(opts.get('instrument', 'Acoustic Grand Piano'), 0),
            scale_filter=sf,
        )
        state['midi_ready'] = True
        state['status'] = 'MIDI ready — download or preview'
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/export')
def export():
    if not pipe.midi_file:
        return jsonify({'error': 'No MIDI generated'}), 400

    buf = io.BytesIO()
    pipe.midi_file.writeFile(buf)
    buf.seek(0)
    name = (state['file_name'] or 'recording').rsplit('.', 1)[0] + '.mid'
    return send_file(buf, mimetype='audio/midi', as_attachment=True, download_name=name)


@app.route('/api/preview')
def preview():
    if not pipe.notes:
        return jsonify({'error': 'No notes'}), 400

    audio = pipe.render_preview()
    buf = io.BytesIO()
    import soundfile as sf
    sf.write(buf, audio, 44100, format='WAV')
    buf.seek(0)
    return send_file(buf, mimetype='audio/wav')


@app.route('/api/original')
def original():
    if pipe.audio_data is None:
        return jsonify({'error': 'No audio'}), 400

    buf = io.BytesIO()
    import soundfile as sf
    sf.write(buf, pipe.audio_data, pipe.sample_rate, format='WAV')
    buf.seek(0)
    return send_file(buf, mimetype='audio/wav')


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Voice to MIDI</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #0f0f0f; color: #e0e0e0; min-height: 100vh; }

.header { background: #1a1a2e; padding: 12px 24px; display: flex; align-items: center;
           gap: 12px; border-bottom: 1px solid #333; }
.header h1 { font-size: 18px; color: #4fc3f7; flex-shrink: 0; }
.header .status { color: #888; font-size: 13px; margin-left: auto; }

.toolbar { background: #1a1a1a; padding: 10px 24px; display: flex; gap: 8px;
            align-items: center; flex-wrap: wrap; border-bottom: 1px solid #252525; }

.btn { padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer;
       font-size: 13px; font-weight: 500; transition: all 0.15s; }
.btn:hover { filter: brightness(1.15); }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-blue { background: #1565c0; color: white; }
.btn-red { background: #c62828; color: white; }
.btn-green { background: #2e7d32; color: white; }
.btn-orange { background: #e65100; color: white; }
.btn-gray { background: #333; color: #ccc; }

.file-label { color: #888; font-size: 13px; margin-left: 8px; }

.main { display: flex; height: calc(100vh - 110px); }

.viz { flex: 1; padding: 16px; display: flex; flex-direction: column; gap: 12px; overflow-y: auto; }

.card { background: #1a1a1a; border-radius: 8px; border: 1px solid #252525; padding: 16px; }
.card h3 { font-size: 13px; color: #888; margin-bottom: 10px; text-transform: uppercase;
            letter-spacing: 0.5px; }

.piano-roll { position: relative; min-height: 200px; overflow-x: auto; }
.note-bar { position: absolute; border-radius: 3px; display: flex; align-items: center;
            justify-content: center; font-size: 10px; color: rgba(255,255,255,0.9);
            font-weight: 500; border: 1px solid rgba(255,255,255,0.15); }
.playhead { position: absolute; top: 0; width: 2px; background: #ff5252;
            z-index: 10; pointer-events: none; box-shadow: 0 0 6px rgba(255,82,82,0.5); }

.sidebar { width: 280px; background: #141414; border-left: 1px solid #252525;
            padding: 16px; overflow-y: auto; flex-shrink: 0; }

.control-group { margin-bottom: 16px; }
.control-group label { display: block; font-size: 12px; color: #888; margin-bottom: 4px; }
.control-group input, .control-group select {
    width: 100%; padding: 6px 10px; background: #222; border: 1px solid #333;
    border-radius: 4px; color: #e0e0e0; font-size: 13px; }
.control-group input[type=range] { padding: 0; }

.range-row { display: flex; align-items: center; gap: 8px; }
.range-row input { flex: 1; }
.range-val { font-size: 12px; color: #4fc3f7; min-width: 32px; text-align: right; }

.section-title { font-size: 11px; color: #555; text-transform: uppercase;
                  letter-spacing: 1px; margin: 16px 0 8px; border-top: 1px solid #252525;
                  padding-top: 12px; }

.note-colors { display: flex; gap: 2px; margin-top: 4px; }

audio { width: 100%; margin-top: 8px; height: 36px; }

.empty-state { display: flex; align-items: center; justify-content: center;
               color: #444; font-size: 14px; flex: 1; }

.note-list { max-height: 300px; overflow-y: auto; }
.note-list table { width: 100%; font-size: 12px; border-collapse: collapse; }
.note-list th { text-align: left; color: #666; padding: 4px 8px; border-bottom: 1px solid #252525; }
.note-list td { padding: 4px 8px; border-bottom: 1px solid #1a1a1a; }
</style>
</head>
<body>

<div class="header">
  <h1>Voice to MIDI</h1>
  <span class="status" id="status">Ready</span>
</div>

<div class="toolbar">
  <label class="btn btn-blue" style="cursor:pointer">
    Open File <input type="file" accept=".wav,.mp3,.flac,.ogg,.m4a" hidden id="fileInput">
  </label>
  <button class="btn btn-red" id="recBtn" onclick="toggleRecord()">Record</button>
  <span class="file-label" id="fileLabel">No file loaded</span>

  <div style="margin-left:auto; display:flex; gap:6px; align-items:center;">
    <button class="btn btn-gray" id="playOrigBtn" onclick="playOrig()" disabled>Play Original</button>
    <button class="btn btn-gray" id="playMidiBtn" onclick="playMidi()" disabled>Play MIDI</button>
    <button class="btn btn-green" id="playBothBtn" onclick="playBoth()" disabled>Play Both</button>
    <button class="btn btn-gray" onclick="stopAudio()">Stop</button>
  </div>
</div>

<div class="main">
  <div class="viz">
    <div class="card" id="waveCard" style="display:none">
      <h3>Audio Waveform</h3>
      <audio id="origAudio" controls style="width:100%"></audio>
    </div>

    <div class="card" style="flex:1; display:flex; flex-direction:column;">
      <h3>Piano Roll <span id="noteCount"></span></h3>
      <div id="pianoRoll" class="piano-roll" style="flex:1;">
        <div class="empty-state">Load an audio file or record from your microphone</div>
      </div>
    </div>

    <div class="card" id="noteListCard" style="display:none">
      <h3>Detected Notes</h3>
      <div class="note-list">
        <table>
          <thead><tr><th>Note</th><th>Start</th><th>End</th><th>Duration</th><th>Velocity</th><th>Confidence</th></tr></thead>
          <tbody id="noteTable"></tbody>
        </table>
      </div>
    </div>
  </div>

  <div class="sidebar">
    <div class="section-title">Detection</div>

    <div class="control-group">
      <label>Min Note Duration (s)</label>
      <div class="range-row">
        <input type="range" id="minDur" min="0.01" max="0.5" step="0.01" value="0.08">
        <span class="range-val" id="minDurVal">0.08</span>
      </div>
    </div>

    <div class="control-group">
      <label>Voiced Threshold</label>
      <div class="range-row">
        <input type="range" id="voicedThresh" min="0" max="1" step="0.05" value="0.5">
        <span class="range-val" id="voicedThreshVal">0.50</span>
      </div>
    </div>

    <div class="control-group">
      <label>Pitch Range</label>
      <div style="display:flex; gap:6px;">
        <select id="fminNote"></select>
        <span style="color:#555">to</span>
        <select id="fmaxNote"></select>
      </div>
    </div>

    <div class="section-title">Quantize</div>

    <div class="control-group">
      <label>Scale</label>
      <select id="scale"></select>
    </div>
    <div class="control-group">
      <label>Root Note</label>
      <select id="rootNote"></select>
    </div>

    <div class="section-title">MIDI Output</div>

    <div class="control-group">
      <label>Tempo (BPM)</label>
      <input type="number" id="tempo" value="120" min="20" max="300">
    </div>
    <div class="control-group">
      <label>Instrument</label>
      <select id="instrument"></select>
    </div>

    <div style="display:flex; flex-direction:column; gap:8px; margin-top:20px;">
      <button class="btn btn-blue" id="detectBtn" onclick="detect()" disabled>Detect Pitch</button>
      <button class="btn btn-green" id="genBtn" onclick="generate()" disabled>Generate MIDI</button>
      <button class="btn btn-orange" id="exportBtn" onclick="exportMidi()" disabled>Export MIDI</button>
    </div>

    <div id="midiAudioWrap" style="margin-top:16px; display:none">
      <label style="font-size:12px; color:#888;">MIDI Preview</label>
      <audio id="midiAudio" controls style="width:100%; margin-top:4px;"></audio>
    </div>
  </div>
</div>

<script>
const INSTRUMENTS = {{ instruments | safe }};
const SCALES = {{ scales | safe }};
const NOTES = {{ notes | safe }};

const NOTE_COLORS = ['#e53935','#ff7043','#ffa726','#ffca28','#66bb6a','#26a69a',
  '#29b6f6','#42a5f5','#5c6bc0','#7e57c2','#ab47bc','#ec407a'];

let pianoRollMaxT = 0;
let pianoRollW = 0;
let animFrameId = null;
let activeAudioSources = [];

// Populate selects
function fillSelect(id, items, def) {
  const sel = document.getElementById(id);
  items.forEach(i => { const o = document.createElement('option'); o.value = i; o.text = i; sel.add(o); });
  if (def) sel.value = def;
}
fillSelect('instrument', INSTRUMENTS, 'Acoustic Grand Piano');
fillSelect('scale', SCALES, 'chromatic');
fillSelect('rootNote', NOTES, 'C');

// Pitch range selects
const noteRange = [];
for (let o = 1; o <= 7; o++) NOTES.forEach(n => noteRange.push(n + o));
fillSelect('fminNote', noteRange, 'C2');
fillSelect('fmaxNote', noteRange, 'C7');

// Slider readouts
document.getElementById('minDur').oninput = e => document.getElementById('minDurVal').textContent = parseFloat(e.target.value).toFixed(2);
document.getElementById('voicedThresh').oninput = e => document.getElementById('voicedThreshVal').textContent = parseFloat(e.target.value).toFixed(2);

function setStatus(s) { document.getElementById('status').textContent = s; }

// File upload
document.getElementById('fileInput').onchange = async function() {
  const f = this.files[0]; if (!f) return;
  const fd = new FormData(); fd.append('file', f);
  setStatus('Loading ' + f.name + '...');
  const r = await fetch('/api/upload', { method: 'POST', body: fd });
  const d = await r.json();
  if (d.error) { setStatus('Error: ' + d.error); return; }
  document.getElementById('fileLabel').textContent = f.name + ' (' + d.duration.toFixed(1) + 's)';
  document.getElementById('detectBtn').disabled = false;
  document.getElementById('playOrigBtn').disabled = false;
  document.getElementById('waveCard').style.display = 'block';
  document.getElementById('origAudio').src = '/api/original?' + Date.now();
  clearPianoRoll();
  setStatus('Loaded: ' + f.name);
};

// Recording
let isRecording = false;
async function toggleRecord() {
  if (!isRecording) {
    const r = await fetch('/api/record/start', { method: 'POST' });
    const d = await r.json();
    if (d.error) { setStatus('Mic error: ' + d.error); return; }
    isRecording = true;
    document.getElementById('recBtn').textContent = 'Stop Rec';
    document.getElementById('recBtn').style.background = '#d50000';
    setStatus('Recording...');
  } else {
    const r = await fetch('/api/record/stop', { method: 'POST' });
    const d = await r.json();
    isRecording = false;
    document.getElementById('recBtn').textContent = 'Record';
    document.getElementById('recBtn').style.background = '#c62828';
    if (d.error) { setStatus(d.error); return; }
    document.getElementById('fileLabel').textContent = 'Recording (' + d.duration.toFixed(1) + 's)';
    document.getElementById('detectBtn').disabled = false;
    document.getElementById('playOrigBtn').disabled = false;
    document.getElementById('waveCard').style.display = 'block';
    document.getElementById('origAudio').src = '/api/original?' + Date.now();
    clearPianoRoll();
    setStatus('Recorded ' + d.duration.toFixed(1) + 's');
  }
}

// Detect
async function detect() {
  setStatus('Detecting pitch...');
  document.getElementById('detectBtn').disabled = true;

  const fminNote = document.getElementById('fminNote').value;
  const fmaxNote = document.getElementById('fmaxNote').value;
  const fmin = noteToFreq(fminNote);
  const fmax = noteToFreq(fmaxNote);

  const r = await fetch('/api/detect', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      min_dur: parseFloat(document.getElementById('minDur').value),
      voiced_thresh: parseFloat(document.getElementById('voicedThresh').value),
      fmin, fmax
    })
  });
  const d = await r.json();
  document.getElementById('detectBtn').disabled = false;
  if (d.error) { setStatus('Error: ' + d.error); return; }

  drawPianoRoll(d.notes);
  fillNoteTable(d.notes);
  document.getElementById('genBtn').disabled = false;
  setStatus('Detected ' + d.count + ' notes');
}

// Generate
async function generate() {
  setStatus('Generating MIDI...');
  const r = await fetch('/api/generate', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      tempo: parseInt(document.getElementById('tempo').value),
      instrument: document.getElementById('instrument').value,
      scale: document.getElementById('scale').value,
      root: document.getElementById('rootNote').value,
    })
  });
  const d = await r.json();
  if (d.error) { setStatus('Error: ' + d.error); return; }
  document.getElementById('exportBtn').disabled = false;
  document.getElementById('playMidiBtn').disabled = false;
  document.getElementById('playBothBtn').disabled = false;
  document.getElementById('midiAudioWrap').style.display = 'block';
  document.getElementById('midiAudio').src = '/api/preview?' + Date.now();
  setStatus('MIDI ready — export or preview');
}

// Export
function exportMidi() {
  window.location.href = '/api/export';
}

// Playback
function playOrig() {
  stopAudio();
  const a = document.getElementById('origAudio');
  a.currentTime = 0; a.play();
  activeAudioSources = [a];
  startPlayhead();
}
function playMidi() {
  stopAudio();
  const a = document.getElementById('midiAudio');
  a.currentTime = 0; a.play();
  activeAudioSources = [a];
  startPlayhead();
}
function playBoth() {
  stopAudio();
  const orig = document.getElementById('origAudio');
  const midi = document.getElementById('midiAudio');
  orig.currentTime = 0; midi.currentTime = 0;
  orig.play(); midi.play();
  activeAudioSources = [orig, midi];
  startPlayhead();
}
function stopAudio() {
  if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
  const orig = document.getElementById('origAudio');
  const midi = document.getElementById('midiAudio');
  orig.pause(); orig.currentTime = 0;
  midi.pause(); midi.currentTime = 0;
  activeAudioSources = [];
  const ph = document.getElementById('playhead');
  if (ph) ph.style.display = 'none';
}

// Playhead animation
function startPlayhead() {
  const ph = document.getElementById('playhead');
  if (!ph || pianoRollMaxT === 0) return;
  ph.style.display = 'block';
  animFrameId = requestAnimationFrame(updatePlayhead);
}
function updatePlayhead() {
  const ph = document.getElementById('playhead');
  if (!ph || activeAudioSources.length === 0) return;

  let currentTime = 0;
  let allEnded = true;
  for (const a of activeAudioSources) {
    if (!a.paused && !a.ended) allEnded = false;
    if (a.currentTime > currentTime) currentTime = a.currentTime;
  }
  if (allEnded) {
    ph.style.display = 'none';
    activeAudioSources = [];
    animFrameId = null;
    return;
  }

  const xPos = Math.min((currentTime / pianoRollMaxT) * pianoRollW, pianoRollW);
  ph.style.left = xPos + 'px';

  // Auto-scroll piano roll to keep playhead visible
  const container = document.getElementById('pianoRoll');
  const visibleRight = container.scrollLeft + container.clientWidth;
  if (xPos > visibleRight - 50 || xPos < container.scrollLeft) {
    container.scrollLeft = Math.max(0, xPos - container.clientWidth * 0.25);
  }

  animFrameId = requestAnimationFrame(updatePlayhead);
}

// Piano roll
function clearPianoRoll() {
  document.getElementById('pianoRoll').innerHTML = '<div class="empty-state">Click "Detect Pitch" to analyze</div>';
  document.getElementById('noteCount').textContent = '';
  document.getElementById('noteListCard').style.display = 'none';
  document.getElementById('genBtn').disabled = true;
  document.getElementById('exportBtn').disabled = true;
  document.getElementById('playMidiBtn').disabled = true;
  document.getElementById('playBothBtn').disabled = true;
  pianoRollMaxT = 0;
  pianoRollW = 0;
}

function drawPianoRoll(notes) {
  const el = document.getElementById('pianoRoll');
  el.innerHTML = '';
  if (!notes.length) { el.innerHTML = '<div class="empty-state">No notes detected</div>'; return; }

  const midis = notes.map(n => n.midi);
  const minM = Math.min(...midis) - 2, maxM = Math.max(...midis) + 2;
  const maxT = Math.max(...notes.map(n => n.end));
  const range = maxM - minM;

  const W = Math.max(el.clientWidth - 20, 600);
  pianoRollMaxT = maxT;
  pianoRollW = W;
  const H = Math.max(range * 18, 120);
  el.style.height = H + 'px';
  el.style.position = 'relative';

  // Grid lines
  for (let m = Math.ceil(minM); m <= Math.floor(maxM); m++) {
    const y = H - ((m - minM) / range) * H;
    const line = document.createElement('div');
    line.style.cssText = `position:absolute; left:0; right:0; top:${y}px; height:1px; background:#222;`;
    if (m % 12 === 0) {
      line.style.background = '#333';
      const lbl = document.createElement('div');
      lbl.style.cssText = `position:absolute; left:2px; top:${y-8}px; font-size:10px; color:#555;`;
      lbl.textContent = noteNameFromMidi(m);
      el.appendChild(lbl);
    }
    el.appendChild(line);
  }

  // Notes
  notes.forEach(n => {
    const x = (n.start / maxT) * W;
    const w = Math.max((n.dur / maxT) * W, 4);
    const y = H - ((n.midi - minM + 0.5) / range) * H - 7;
    const bar = document.createElement('div');
    bar.className = 'note-bar';
    bar.style.left = x + 'px';
    bar.style.top = y + 'px';
    bar.style.width = w + 'px';
    bar.style.height = '14px';
    bar.style.background = NOTE_COLORS[n.midi % 12];
    bar.style.opacity = 0.5 + n.conf * 0.5;
    bar.textContent = n.name;
    bar.title = `${n.name} | ${n.start.toFixed(2)}s-${n.end.toFixed(2)}s | vel:${n.vel}`;
    el.appendChild(bar);
  });

  const playhead = document.createElement('div');
  playhead.className = 'playhead';
  playhead.id = 'playhead';
  playhead.style.left = '0px';
  playhead.style.height = H + 'px';
  playhead.style.display = 'none';
  el.appendChild(playhead);

  document.getElementById('noteCount').textContent = '(' + notes.length + ' notes)';
}

function fillNoteTable(notes) {
  const tb = document.getElementById('noteTable');
  tb.innerHTML = notes.map(n =>
    `<tr><td style="color:${NOTE_COLORS[n.midi%12]}">${n.name}</td><td>${n.start.toFixed(2)}s</td>` +
    `<td>${n.end.toFixed(2)}s</td><td>${n.dur.toFixed(2)}s</td><td>${n.vel}</td>` +
    `<td>${(n.conf*100).toFixed(0)}%</td></tr>`
  ).join('');
  document.getElementById('noteListCard').style.display = notes.length ? 'block' : 'none';
}

function noteNameFromMidi(m) {
  const names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
  return names[m % 12] + (Math.floor(m / 12) - 1);
}

function noteToFreq(name) {
  const notes = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11};
  const match = name.match(/^([A-G]#?)(\d+)$/);
  if (!match) return 440;
  const midi = (parseInt(match[2]) + 1) * 12 + notes[match[1]];
  return 440 * Math.pow(2, (midi - 69) / 12);
}

// Clean up playhead when audio ends naturally
document.getElementById('origAudio').addEventListener('ended', onAudioEnded);
document.getElementById('midiAudio').addEventListener('ended', onAudioEnded);
function onAudioEnded(e) {
  activeAudioSources = activeAudioSources.filter(a => a !== e.target);
  if (activeAudioSources.length === 0) {
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
    const ph = document.getElementById('playhead');
    if (ph) ph.style.display = 'none';
  }
}
</script>
</body>
</html>
'''
