"""
audio_to_notes.py — Transcription router.

Selects the transcription backend and exposes a single `transcribe_audio()`
function used by the rest of the pipeline (server.py, comparator, etc.).

Backends
--------
  "bytedance"   ByteDance High-Resolution Piano (~96.7% F1, best quality)
                Requires: pip install piano_transcription_inference torch
                Model:    ~370 MB download on first use

  "basicpitch"  Spotify Basic Pitch ONNX (~72-81% F1, fast & lightweight)
                Requires: pip install basic-pitch onnxruntime
                No additional download needed

The default is "bytedance" with automatic fallback to "basicpitch" if the
ByteDance package is not installed.

Re-exports
----------
  Note, midi_to_notes, notes_to_midi  (from note_model)
  transcribe_audio                    (router)
"""
from __future__ import annotations

from typing import List, Optional

# Re-export shared types so existing imports keep working
from .note_model import Note, midi_to_notes, notes_to_midi  # noqa: F401

# Default backend — change to "basicpitch" to always use Basic Pitch
DEFAULT_BACKEND = "bytedance"


def transcribe_audio(
    audio_path: str,
    backend: Optional[str] = None,
    ref_notes: Optional[List[Note]] = None,
    # Basic Pitch specific (ignored for ByteDance)
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 0.05,
    suppress_harmonics: bool = True,
    # ByteDance specific (ignored for Basic Pitch)
    device: str = "cpu",
) -> List[Note]:
    """
    Transcribe an audio file to a list of Note objects.

    Args:
        audio_path:          Path to audio file (WAV, MP3, FLAC).
        backend:             "bytedance" | "basicpitch" | None (auto-detect).
                             None → tries ByteDance first, falls back to Basic Pitch.
        ref_notes:           Reference MIDI notes. Used by Basic Pitch for ref-aware
                             harmonic suppression. Ignored by ByteDance (not needed).
        onset_threshold:     [Basic Pitch] onset detection threshold (0-1).
        frame_threshold:     [Basic Pitch] frame detection threshold (0-1).
        minimum_note_length: [Basic Pitch] minimum note duration in seconds.
        suppress_harmonics:  [Basic Pitch] apply dedup + harmonic post-processing.
        device:              [ByteDance] "cpu" or "cuda".

    Returns:
        Sorted list of Note objects, same format regardless of backend.
    """
    chosen = (backend or DEFAULT_BACKEND).lower()

    # Auto-fallback: ByteDance requested but not installed → use Basic Pitch
    if chosen == "bytedance":
        from . import transcribe_bytedance as _bd
        if not _bd.is_available():
            print("  [Router] ByteDance not installed — falling back to Basic Pitch.")
            print("  [Router] To install: pip install piano_transcription_inference torch")
            chosen = "basicpitch"

    if chosen == "bytedance":
        from . import transcribe_bytedance as _bd
        return _bd.transcribe(audio_path, device=device)

    elif chosen == "basicpitch":
        from . import transcribe_basicpitch as _bp
        return _bp.transcribe(
            audio_path,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            suppress_harmonics=suppress_harmonics,
            ref_notes=ref_notes,
        )

    else:
        raise ValueError(
            f"Unknown backend: {chosen!r}. Choose 'bytedance' or 'basicpitch'."
        )