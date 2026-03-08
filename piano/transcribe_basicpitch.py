"""
transcribe_basicpitch.py — Audio → Notes using Spotify Basic Pitch (ONNX backend).

Accuracy:  ~72–81% F1 on MAESTRO (polyphonic piano)
Speed:     Fast (~1× realtime on CPU)
Install:   pip install basic-pitch onnxruntime

Strengths
  • Lightweight, runs entirely on CPU
  • Good on monophonic / simple melodies
  • No model download required at runtime

Weaknesses
  • Produces harmonic false positives (attack transients, overtones)
  • Lower accuracy on dense polyphonic chords
  • Requires post-processing (dedup + harmonic suppression) for clean output
"""
from __future__ import annotations

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from typing import List, Optional
from pathlib import Path

from .note_model import Note, PIANO_MIDI_MIN, PIANO_MIDI_MAX


# ── Harmonic suppression ──────────────────────────────────────────────────────

_HARMONIC_INTERVALS_UP   = {12, 19, 24, 28, 31, 33, 36}  # 2nd–8th harmonic partial
_HARMONIC_INTERVALS_DOWN = {12, 24, 36}                   # sub-harmonics


def _build_harmonic_set(notes: List[Note]) -> set:
    """Return the set of MIDI pitches that are acoustic harmonics of the given notes."""
    pitches = set(n.pitch for n in notes)
    harmonics: set = set()
    for p in pitches:
        for iv in _HARMONIC_INTERVALS_UP:
            h = p + iv
            if PIANO_MIDI_MIN <= h <= PIANO_MIDI_MAX:
                harmonics.add(h)
        for iv in _HARMONIC_INTERVALS_DOWN:
            h = p - iv
            if PIANO_MIDI_MIN <= h <= PIANO_MIDI_MAX:
                harmonics.add(h)
    return harmonics - pitches  # never suppress notes that are themselves real


def _suppress_harmonics(
    perf_notes: List[Note],
    ref_notes:  Optional[List[Note]] = None,
    amplitude_threshold: float = 0.0,
) -> List[Note]:
    """
    Three-pass post-processing to clean up Basic Pitch output.

    Pass 1 — Deduplicate: same pitch within 60 ms → keep the louder detection.
              Basic Pitch fires on both the attack transient and the sustain.

    Pass 2 — Ref-aware harmonic suppression: build the harmonic series from the
              *reference* pitches and drop any perf note that is a harmonic of a
              nearby ref note and is quieter than that ref note.

    Pass 3 — Hard amplitude floor (optional; use 0.55 for synth, 0.0 for acoustic).
    """
    if not perf_notes:
        return perf_notes

    # ── Pass 1: deduplicate ───────────────────────────────────────────────────
    notes = sorted(perf_notes, key=lambda n: (n.pitch, n.start))
    deduped: List[Note] = []
    skip: set = set()
    for i, n in enumerate(notes):
        if i in skip:
            continue
        best = n
        for j in range(i + 1, len(notes)):
            other = notes[j]
            if other.pitch != n.pitch or other.start - n.start > 0.06:
                break
            skip.add(j)
            if other.velocity > best.velocity:
                best = other
        deduped.append(best)
    notes = sorted(deduped, key=lambda n: (n.start, n.pitch))

    # ── Pass 2: ref-aware harmonic suppression ────────────────────────────────
    if ref_notes:
        harmonic_pitches = _build_harmonic_set(ref_notes)
        ref_sorted = sorted(ref_notes, key=lambda n: n.start)
        result: List[Note] = []
        for n in notes:
            if n.pitch in harmonic_pitches:
                nearby = [r for r in ref_sorted if abs(r.start - n.start) <= 0.15]
                if nearby:
                    strongest = max(nearby, key=lambda r: r.velocity)
                    if n.velocity <= strongest.velocity:
                        continue  # suppress — weaker than the fundamental that caused it
            result.append(n)
        notes = result

    # ── Pass 3: amplitude floor ───────────────────────────────────────────────
    if amplitude_threshold > 0:
        notes = [n for n in notes if n.velocity >= amplitude_threshold * 127]

    return sorted(notes, key=lambda n: (n.start, n.pitch))


# ── ONNX model discovery ──────────────────────────────────────────────────────

def _find_onnx_model() -> str:
    from basic_pitch import ICASSP_2022_MODEL_PATH
    model_dir = Path(ICASSP_2022_MODEL_PATH)
    onnx_path = model_dir.parent / (model_dir.name + ".onnx")
    if onnx_path.exists():
        return str(onnx_path)
    for f in model_dir.parent.rglob("*.onnx"):
        return str(f)
    return str(ICASSP_2022_MODEL_PATH)


# ── Public API ────────────────────────────────────────────────────────────────

def transcribe(
    audio_path: str,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 0.05,
    melodia_trick: bool = True,
    suppress_harmonics: bool = True,
    ref_notes: Optional[List[Note]] = None,
) -> List[Note]:
    """
    Transcribe audio to notes using Spotify Basic Pitch (ONNX).

    Args:
        audio_path:        Path to WAV/MP3/FLAC file.
        onset_threshold:   Confidence threshold for note onsets (0–1).
        frame_threshold:   Confidence threshold for note frames (0–1).
        minimum_note_length: Minimum note duration in seconds.
        melodia_trick:     Use melodia post-processing for melody extraction.
        suppress_harmonics: Apply dedup + harmonic suppression post-processing.
        ref_notes:         Reference MIDI notes for ref-aware harmonic suppression.
                           Pass these for best accuracy when comparing against a known MIDI.

    Returns:
        Sorted list of Note objects.
    """
    from basic_pitch.inference import predict

    model_path = _find_onnx_model()
    print(f"  [BasicPitch] Transcribing: {Path(audio_path).name}")
    print(f"  Backend: ONNX  |  Model: {Path(model_path).name}")

    _, _, note_events = predict(
        audio_path,
        model_or_model_path=model_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
        melodia_trick=melodia_trick,
        midi_tempo=120,
    )

    notes: List[Note] = []
    for event in note_events:
        if isinstance(event, dict):
            pitch     = int(event["pitch_midi"])
            start     = float(event["start_time_s"])
            end       = float(event["end_time_s"])
            amplitude = float(event["amplitude"])
        else:
            start, end, pitch, amplitude = (
                float(event[0]), float(event[1]), int(event[2]), float(event[3])
            )

        if not (PIANO_MIDI_MIN <= pitch <= PIANO_MIDI_MAX):
            continue

        notes.append(Note(
            pitch=pitch,
            start=start,
            end=end,
            velocity=int(np.clip(amplitude * 127, 1, 127)),
            frequency=440.0 * (2 ** ((pitch - 69) / 12)),
        ))

    notes.sort(key=lambda n: (n.start, n.pitch))
    raw_count = len(notes)

    if suppress_harmonics:
        notes = _suppress_harmonics(notes, ref_notes=ref_notes)
        print(f"  Detected {raw_count} notes → {len(notes)} after harmonic suppression.")
    else:
        print(f"  Detected {raw_count} notes.")

    return notes