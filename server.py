"""
server.py — AI Music Teacher web server
────────────────────────────────────────
Run with:  python server.py
Then open: http://localhost:8000
"""

import os
import uuid
import json
import traceback
import threading
import urllib.parse
import re
import io
from pathlib import Path
from typing import Optional

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from http.server import HTTPServer, BaseHTTPRequestHandler

from piano import transcribe_audio, midi_to_notes, compare_notes
from piano.visualizer import draw_piano_roll, draw_timing_deviation_chart, draw_accuracy_by_pitch_class

UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("./output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job store: { job_id: { status, result, error } }
jobs: dict = {}


def run_analysis_job(job_id: str, midi_path: str, audio_path: Optional[str], midi2_path: Optional[str], backend: str = "bytedance"):
    """Background thread: run the full comparison pipeline."""
    try:
        jobs[job_id]["status"] = "loading"

        ref_notes = midi_to_notes(midi_path)
        if not ref_notes:
            raise ValueError("No notes found in reference MIDI file.")

        if audio_path:
            jobs[job_id]["status"] = "transcribing"
            perf_notes = transcribe_audio(audio_path, backend=backend, ref_notes=ref_notes)
        else:
            jobs[job_id]["status"] = "loading"
            perf_notes = midi_to_notes(midi2_path)

        if not perf_notes:
            raise ValueError("No notes found in performance file.")

        jobs[job_id]["status"] = "comparing"
        result = compare_notes(ref_notes, perf_notes, ref_bpm=120.0, dtw_align=True, amplitude_threshold=0.0)

        # Save visuals
        job_out = OUTPUT_DIR / job_id
        job_out.mkdir(exist_ok=True)

        draw_piano_roll(result, str(job_out / "piano_roll.png"), title="Piano Roll Comparison")
        draw_timing_deviation_chart(result, str(job_out / "timing_deviation.png"))
        draw_accuracy_by_pitch_class(result, str(job_out / "pitch_class.png"))

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = {
            "accuracy":    round(result.accuracy * 100, 2),
            "precision":   round(result.precision * 100, 2),
            "recall":      round(result.recall * 100, 2),
            "f1":          round(result.f1 * 100, 2),
            "n_correct":      result.n_correct,
            "n_early":        result.n_early,
            "n_late":         result.n_late,
            "n_missing":      result.n_missing,
            "n_extra":        result.n_extra,
            "n_wrong_pitch":  result.n_wrong_pitch,
            "mean_onset_dev_ms":    round(result.mean_onset_dev_ms, 1),
            "std_onset_dev_ms":     round(result.std_onset_dev_ms, 1),
            "mean_pitch_dev_cents": round(result.mean_pitch_dev_cents, 2),
            "tempo": {
                "overall_ratio":    round(result.tempo.overall_ratio, 3),
                "bpm_ref":          round(result.tempo.overall_bpm_ref, 1),
                "bpm_perf":         round(result.tempo.overall_bpm_perf, 1),
                "verdict":          result.tempo.verdict,
                "local_ratios":     [round(r, 3) for r in result.tempo.local_ratios],
            },
            "dynamics": {
                "mean_velocity_ratio": round(result.dynamics.mean_velocity_ratio, 2),
                "std_velocity_ratio":  round(result.dynamics.std_velocity_ratio, 2),
                "verdict":             result.dynamics.verdict,
            },
            "per_pitch": result.per_pitch,
            "job_id": job_id,
        }

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        traceback.print_exc()


# ── Multipart parser (no cgi module — works on Python 3.13+) ──────────────────

def parse_multipart(handler):
    """
    Parse multipart/form-data manually.
    Returns dict: { field_name: { 'filename': str|None, 'data': bytes } }
    """
    content_type = handler.headers.get("Content-Type", "")
    content_length = int(handler.headers.get("Content-Length", 0))
    body = handler.rfile.read(content_length)

    # Extract boundary
    boundary_match = re.search(r'boundary=([^\s;]+)', content_type)
    if not boundary_match:
        raise ValueError("No boundary found in Content-Type")
    boundary = boundary_match.group(1).strip('"').encode()

    fields = {}
    # Split on --boundary
    delimiter = b'--' + boundary
    parts = body.split(delimiter)

    for part in parts:
        if not part or part == b'--\r\n' or part == b'--':
            continue
        # Strip leading CRLF
        if part.startswith(b'\r\n'):
            part = part[2:]
        if part.endswith(b'\r\n'):
            part = part[:-2]

        # Split headers from body
        if b'\r\n\r\n' not in part:
            continue
        raw_headers, content = part.split(b'\r\n\r\n', 1)

        headers = {}
        for line in raw_headers.decode('utf-8', errors='replace').splitlines():
            if ':' in line:
                k, v = line.split(':', 1)
                headers[k.strip().lower()] = v.strip()

        disposition = headers.get('content-disposition', '')
        name_match = re.search(r'name="([^"]+)"', disposition)
        file_match = re.search(r'filename="([^"]*)"', disposition)

        if not name_match:
            continue

        name     = name_match.group(1)
        filename = file_match.group(1) if file_match else None
        fields[name] = {'filename': filename, 'data': content}

    return fields


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default access logs

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, mime: str):
        if not path.exists():
            self.send_json({"error": "File not found"}, 404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", len(data))
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/" or path == "/index.html":
            html = (Path(__file__).parent / "static" / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", len(html))
            self.end_headers()
            self.wfile.write(html)

        elif path.startswith("/status/"):
            job_id = path.split("/status/")[1]
            if job_id not in jobs:
                self.send_json({"error": "Job not found"}, 404)
            else:
                self.send_json(jobs[job_id])

        elif path.startswith("/image/"):
            parts = path.split("/")[2:]  # [job_id, filename]
            if len(parts) == 2:
                img_path = OUTPUT_DIR / parts[0] / parts[1].split("?")[0]
                self.send_file(img_path, "image/png")
            else:
                self.send_json({"error": "Invalid path"}, 400)

        else:
            self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/analyze":
            try:
                form = parse_multipart(self)

                if "midi" not in form:
                    self.send_json({"error": "Missing midi field"}, 400)
                    return

                job_id = str(uuid.uuid4())[:8]
                jobs[job_id] = {"status": "queued", "result": None, "error": None}

                # Save reference MIDI
                midi_path = str(UPLOAD_DIR / f"{job_id}_ref.mid")
                with open(midi_path, "wb") as f:
                    f.write(form["midi"]["data"])

                audio_path = None
                midi2_path = None

                if "audio" in form and form["audio"].get("filename"):
                    ext = Path(form["audio"]["filename"]).suffix or ".wav"
                    audio_path = str(UPLOAD_DIR / f"{job_id}_perf{ext}")
                    with open(audio_path, "wb") as f:
                        f.write(form["audio"]["data"])
                elif "midi2" in form and form["midi2"].get("filename"):
                    midi2_path = str(UPLOAD_DIR / f"{job_id}_perf.mid")
                    with open(midi2_path, "wb") as f:
                        f.write(form["midi2"]["data"])
                else:
                    self.send_json({"error": "Provide audio or second MIDI file"}, 400)
                    return

                # Backend selection: 'bytedance' (default) or 'basicpitch'
                backend = 'basicpitch' if form.get('backend', {}).get('data', b'').decode() == 'basicpitch' else 'bytedance'

                # Run in background thread
                t = threading.Thread(
                    target=run_analysis_job,
                    args=(job_id, midi_path, audio_path, midi2_path),
                    kwargs={'backend': backend},
                    daemon=True,
                )
                t.start()

                self.send_json({"job_id": job_id, "status": "queued"})

            except Exception as e:
                self.send_json({"error": str(e)}, 500)
                traceback.print_exc()
        else:
            self.send_json({"error": "Not found"}, 404)


if __name__ == "__main__":
    port = 8000
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"\n  🎹  AI Music Teacher")
    print(f"  Running at → http://localhost:{port}\n")
    server.serve_forever()