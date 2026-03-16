"""
Compatibility wrapper for the removed Basic Pitch backend.

Internal callers no longer use this module, but keeping the import path avoids
breaking older scripts that still reference `piano.transcribe_basicpitch`.
"""
from __future__ import annotations

from typing import List, Optional

from .note_model import Note


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
    Compatibility wrapper that delegates to the ByteDance backend.
    """
    del onset_threshold
    del frame_threshold
    del minimum_note_length
    del melodia_trick
    del suppress_harmonics
    del ref_notes

    print("  [BasicPitch] Basic Pitch support was removed; using ByteDance instead.")

    from . import transcribe_bytedance as _bd

    return _bd.transcribe(audio_path)
