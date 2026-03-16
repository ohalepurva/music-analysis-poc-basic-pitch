"""
audio_to_notes.py — Transcription router.

Routes transcription through the ByteDance piano backend and exposes a single
`transcribe_audio()` function used by the rest of the pipeline.

`backend="basicpitch"` is still accepted as a legacy alias so older callers do
not fail after removing the Basic Pitch dependency.

Re-exports
----------
  Note, midi_to_notes, notes_to_midi  (from note_model)
  transcribe_audio                    (router)
"""
from __future__ import annotations

from typing import List, Optional

# Re-export shared types so existing imports keep working
from .note_model import Note, midi_to_notes, notes_to_midi  # noqa: F401

DEFAULT_BACKEND = "bytedance"


def transcribe_audio(
    audio_path: str,
    backend: Optional[str] = None,
    ref_notes: Optional[List[Note]] = None,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 0.05,
    suppress_harmonics: bool = True,
    device: str = "cpu",
) -> List[Note]:
    """
    Transcribe an audio file to a list of Note objects.

    Args:
        audio_path:          Path to audio file (WAV, MP3, FLAC).
        backend:             Legacy backend hint. "basicpitch" is mapped to
                             "bytedance" for compatibility.
        ref_notes:           Unused compatibility argument.
        onset_threshold:     Unused compatibility argument.
        frame_threshold:     Unused compatibility argument.
        minimum_note_length: Unused compatibility argument.
        suppress_harmonics:  Unused compatibility argument.
        device:              ByteDance device, usually "cpu" or "cuda".

    Returns:
        Sorted list of Note objects.
    """
    chosen = (backend or DEFAULT_BACKEND).lower()

    if chosen == "basicpitch":
        print("  [Router] Basic Pitch support was removed; using ByteDance instead.")
        chosen = DEFAULT_BACKEND

    if chosen != DEFAULT_BACKEND:
        raise ValueError(
            f"Unknown backend: {chosen!r}. Choose 'bytedance'."
        )

    from . import transcribe_bytedance as _bd

    return _bd.transcribe(audio_path, device=device)
