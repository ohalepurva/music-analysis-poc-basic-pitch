"""
note_model.py — Shared Note dataclass and MIDI I/O utilities.

Imported by all transcription backends so the rest of the pipeline
(comparator, visualizer, server) never needs to know which backend ran.
"""
from __future__ import annotations

import numpy as np
import pretty_midi
from dataclasses import dataclass
from typing import List
from pathlib import Path

PIANO_MIDI_MIN = 21   # A0
PIANO_MIDI_MAX = 108  # C8


@dataclass
class Note:
    pitch:     int    # MIDI note number (21–108)
    start:     float  # seconds
    end:       float  # seconds
    velocity:  int    # 1–127
    frequency: float  # Hz (always pure 12-TET, not bent)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def midi_frequency(self) -> float:
        """Theoretical 12-TET frequency for this MIDI pitch."""
        return 440.0 * (2 ** ((self.pitch - 69) / 12))

    @property
    def cents_deviation(self) -> float:
        """Deviation of self.frequency from 12-TET in cents."""
        if self.frequency <= 0:
            return 0.0
        return 1200 * np.log2(self.frequency / self.midi_frequency)

    def __repr__(self) -> str:
        name = pretty_midi.note_number_to_name(self.pitch)
        return f"Note({name}, {self.start:.3f}s–{self.end:.3f}s, vel={self.velocity})"


# ── MIDI I/O ──────────────────────────────────────────────────────────────────

def midi_to_notes(midi_path: str) -> List[Note]:
    """Parse a MIDI file into a sorted list of Note objects."""
    print(f"  Loading MIDI: {Path(midi_path).name}")
    pm = pretty_midi.PrettyMIDI(midi_path)

    notes: List[Note] = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for n in instrument.notes:
            if not (PIANO_MIDI_MIN <= n.pitch <= PIANO_MIDI_MAX):
                continue
            notes.append(Note(
                pitch=int(n.pitch),
                start=float(n.start),
                end=float(n.end),
                velocity=int(n.velocity),
                frequency=440.0 * (2 ** ((n.pitch - 69) / 12)),
            ))

    notes.sort(key=lambda n: (n.start, n.pitch))
    print(f"  Loaded {len(notes)} notes from MIDI.")
    return notes


def notes_to_midi(notes: List[Note], output_path: str, tempo: float = 120.0) -> None:
    """Save a list of Notes to a MIDI file."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0, name="Piano")
    for n in notes:
        piano.notes.append(pretty_midi.Note(
            velocity=n.velocity,
            pitch=n.pitch,
            start=n.start,
            end=max(n.end, n.start + 0.03),
        ))
    pm.instruments.append(piano)
    pm.write(output_path)
    print(f"  Saved MIDI → {output_path}")