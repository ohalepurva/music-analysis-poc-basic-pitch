"""
transcribe_bytedance.py — Audio → Notes using ByteDance High-Resolution Piano Transcription.

Accuracy:  ~96.7% F1 on MAESTRO (vs Basic Pitch ~72%)
Speed:     ~2–5× realtime on CPU (GPU recommended for long files)
Install:   pip install piano_transcription_inference torch

Model:     Pretrained weights auto-downloaded on first run (~370 MB)
           Cached at: ~/.piano_transcription_inference/

Paper:     "High-resolution Piano Transcription with Pedals by
            Regressing Onsets and Offsets Times"
           Kong et al., 2021  (ByteDance AI Lab / qiuqiangkong)

Strengths
  • Best open-source accuracy on polyphonic piano
  • Predicts onsets, offsets, velocity AND sustain pedal via regression
  • No harmonic post-processing needed — regression avoids attack artifacts
  • Handles dense chords, wide dynamic range, sustain pedal resonance

Weaknesses
  • Larger model (~370 MB download on first use)
  • Slower on CPU than Basic Pitch
  • Piano-only (not designed for other instruments)
"""
from __future__ import annotations

from typing import List, Optional
from pathlib import Path

from .note_model import Note, PIANO_MIDI_MIN, PIANO_MIDI_MAX


def _check_installed() -> None:
    try:
        import piano_transcription_inference  # noqa: F401
    except ImportError:
        raise ImportError(
            "ByteDance backend not installed.\n"
            "Run:  pip install piano_transcription_inference torch\n"
            "The pretrained model (~370 MB) will be downloaded on first use."
        )


def transcribe(
    audio_path: str,
    device: str = "cpu",
) -> List[Note]:
    """
    Transcribe audio to notes using the ByteDance High-Resolution Piano model.

    Args:
        audio_path: Path to WAV/MP3/FLAC file.
        device:     'cpu' or 'cuda' (use 'cuda' for GPU acceleration).

    Returns:
        Sorted list of Note objects.

    No post-processing is applied — the ByteDance model's regression-based
    architecture avoids the harmonic artifacts that require suppression in
    frame-classification models like Basic Pitch.
    """
    _check_installed()

    import tempfile, os, numpy as np
    from piano_transcription_inference import PianoTranscription, sample_rate

    print(f"  [ByteDance] Transcribing: {Path(audio_path).name}")
    print(f"  Device: {device}")

    # Load audio at the model's required sample rate (16 kHz).
    # Bypass ByteDance's load_audio (uses audioread which needs ffmpeg installed).
    # Try librosa first (handles MP3/FLAC/WAV), fall back to soundfile (WAV only).
    audio = None
    try:
        import librosa
        audio, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    except Exception:
        pass

    if audio is None:
        try:
            import soundfile as sf
            from scipy.signal import resample_poly
            from math import gcd
            raw, orig_sr = sf.read(audio_path, always_2d=True)
            raw = raw.mean(axis=1)  # mono
            if orig_sr != sample_rate:
                g = gcd(sample_rate, orig_sr)
                audio = resample_poly(raw, sample_rate // g, orig_sr // g).astype(np.float32)
            else:
                audio = raw.astype(np.float32)
        except Exception as e:
            raise RuntimeError(
                f"Could not load audio file '{audio_path}'.\n"
                f"Install librosa:  pip install librosa\n"
                f"Or ffmpeg:        brew install ffmpeg\n"
                f"Original error:   {e}"
            )

    # ── Compatibility patches for the installed package ───────────────────────
    # 1. PyTorch 2.6 changed torch.load default to weights_only=True, which
    #    breaks loading of old-style checkpoints. Patch it back to False.
    # 2. The package uses `wget` to download the model, which isn't on macOS
    #    by default. Patch it to use `curl` instead.
    import piano_transcription_inference.inference as _pti
    import torch as _torch

    _original_torch_load = _torch.load
    def _patched_torch_load(f, *args, **kwargs):
        kwargs.setdefault('weights_only', False)
        return _original_torch_load(f, *args, **kwargs)
    _torch.load = _patched_torch_load

    # Patch wget → curl for macOS
    import os as _os
    _original_system = _os.system
    def _patched_system(cmd):
        if 'wget' in cmd:
            cmd = cmd.replace('wget -O', 'curl -L -o')
        return _original_system(cmd)
    _os.system = _patched_system

    # Run transcription — outputs a MIDI file
    transcriptor = PianoTranscription(device=device)

    # Restore originals
    _torch.load = _original_torch_load
    _os.system  = _original_system

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_midi = tmp.name

    try:
        transcriptor.transcribe(audio, tmp_midi)

        # Parse the output MIDI into Note objects
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(tmp_midi)
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
    finally:
        os.unlink(tmp_midi)

    notes.sort(key=lambda n: (n.start, n.pitch))
    print(f"  Detected {len(notes)} notes.")
    return notes


def is_available() -> bool:
    """Return True if the ByteDance backend is installed and importable."""
    try:
        import piano_transcription_inference  # noqa: F401
        return True
    except ImportError:
        return False