"""
comparator.py — Robust piano performance analysis engine

Design philosophy:
  - Transcription is imperfect, especially for acoustic piano.
    The comparator must be tolerant and give meaningful feedback
    even at 75-85% transcription accuracy.

  - Uses DTW to align student tempo against reference before note matching.
    A student who plays 10% slower should not get penalised on every note.

  - Per-note status is one of:
      correct       — right pitch, on time
      early         — right pitch, played too early (> 80ms)
      late          — right pitch, played too late  (> 80ms)
      missing       — note not detected in performance
      wrong_pitch   — something played at the right time but wrong note
      extra         — played something not in the score

  - Feedback dimensions:
      accuracy      — % of reference notes correctly played
      timing        — mean/std onset deviation in ms
      tempo         — overall speed ratio vs reference
      dynamics      — mean velocity ratio (loud/soft vs score)
"""

from __future__ import annotations
import numpy as np
import mir_eval
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from .note_model import Note

# ── Tolerances ─────────────────────────────────────────────────────────────────
CORRECT_ONSET_MS   =  80   # ±80ms  → "correct" timing
LATE_EARLY_ONSET_MS = 200  # ±200ms → still counts as the right note, just early/late
PITCH_SEMITONES    =  1    # ±1 semitone for pitch matching
MIR_ONSET_TOL      =  0.1  # 100ms for mir_eval F1
MIR_PITCH_TOL      = 50.0  # 50 cents for mir_eval (Hz input)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class NoteMatch:
    ref_note:        Optional[Note]
    perf_note:       Optional[Note]
    status:          str    # correct | early | late | missing | wrong_pitch | extra
    onset_dev_ms:    float  # positive = late
    pitch_dev_cents: float
    velocity_ratio:  float  # perf_velocity / ref_velocity  (1.0 = perfect dynamics)


@dataclass
class TempoAnalysis:
    overall_ratio:    float   # perf_duration / ref_duration  (<1 = faster, >1 = slower)
    overall_bpm_ref:  float
    overall_bpm_perf: float
    local_ratios:     List[float]   # per-section tempo ratios
    verdict:          str    # "on tempo" | "too fast" | "too slow" | "uneven"


@dataclass
class DynamicsAnalysis:
    mean_velocity_ratio: float   # >1 = louder than score, <1 = softer
    std_velocity_ratio:  float
    soft_sections:       List[Tuple[float, float]]   # (start, end) of too-soft passages
    loud_sections:       List[Tuple[float, float]]   # (start, end) of too-loud passages
    verdict:             str     # "good" | "too soft" | "too loud" | "uneven"


@dataclass
class ComparisonResult:
    # Core
    matches:           List[NoteMatch]
    accuracy:          float   # F1 score (mir_eval)
    precision:         float
    recall:            float
    f1:                float

    # Note counts
    n_correct:         int
    n_early:           int
    n_late:            int
    n_missing:         int
    n_wrong_pitch:     int
    n_extra:           int

    # Timing (ms)
    mean_onset_dev_ms: float
    std_onset_dev_ms:  float

    # Pitch deviation (cents) — meaningful for acoustic piano
    mean_pitch_dev_cents: float

    # Tempo
    tempo:             TempoAnalysis

    # Dynamics
    dynamics:          DynamicsAnalysis

    # Per pitch class accuracy
    per_pitch:         Dict[str, float]

    def summary(self) -> str:
        t = self.tempo
        d = self.dynamics
        lines = [
            "=" * 62,
            "         PIANO PERFORMANCE ANALYSIS",
            "=" * 62,
            f"  Accuracy (F1):          {self.accuracy*100:6.1f}%",
            f"  Precision:              {self.precision*100:6.1f}%",
            f"  Recall:                 {self.recall*100:6.1f}%",
            "",
            "  ── Notes ────────────────────────────────────────",
            f"  Correct:                {self.n_correct}",
            f"  Early  (>{CORRECT_ONSET_MS}ms):       {self.n_early}",
            f"  Late   (>{CORRECT_ONSET_MS}ms):       {self.n_late}",
            f"  Missing:                {self.n_missing}",
            f"  Wrong pitch:            {self.n_wrong_pitch}",
            f"  Extra:                  {self.n_extra}",
            "",
            "  ── Timing ───────────────────────────────────────",
            f"  Mean onset deviation:   {self.mean_onset_dev_ms:+.0f} ms",
            f"  Std onset deviation:    {self.std_onset_dev_ms:.0f} ms",
            "",
            "  ── Tempo ────────────────────────────────────────",
            f"  Reference BPM:          {t.overall_bpm_ref:.1f}",
            f"  Performed BPM:          {t.overall_bpm_perf:.1f}",
            f"  Speed ratio:            {t.overall_ratio:.2f}x  ({t.verdict})",
            "",
            "  ── Dynamics ─────────────────────────────────────",
            f"  Velocity ratio:         {d.mean_velocity_ratio:.2f}  ({d.verdict})",
            "=" * 62,
        ]
        return "\n".join(lines)


# ── DTW tempo alignment ────────────────────────────────────────────────────────

def _dtw_align(ref_notes: List[Note], perf_notes: List[Note]) -> List[Note]:
    """
    Warp perf note timestamps to best match ref using DTW on onset sequences.
    Returns perf_notes with adjusted start/end times.
    """
    if len(perf_notes) < 4 or len(ref_notes) < 4:
        return perf_notes

    ref_onsets  = np.array([n.start for n in ref_notes])
    perf_onsets = np.array([n.start for n in perf_notes])

    # Normalise both to [0,1] for DTW
    ref_norm  = (ref_onsets  - ref_onsets[0])  / max(ref_onsets[-1]  - ref_onsets[0],  1e-6)
    perf_norm = (perf_onsets - perf_onsets[0]) / max(perf_onsets[-1] - perf_onsets[0], 1e-6)

    try:
        from dtaidistance import dtw as dtw_lib
        _, paths = dtw_lib.warping_paths(ref_norm.astype(float),
                                          perf_norm.astype(float))
        path = dtw_lib.best_path(paths)
        # Build a piecewise-linear mapping: perf_time → ref_time
        ref_times_matched  = [ref_onsets[r]  for r, p in path]
        perf_times_matched = [perf_onsets[p] for r, p in path]

        # Apply mapping to all perf notes
        aligned = []
        for note in perf_notes:
            new_start = float(np.interp(note.start, perf_times_matched, ref_times_matched))
            new_end   = new_start + note.duration
            aligned.append(Note(note.pitch, new_start, new_end,
                                note.velocity, note.frequency))
        return aligned
    except Exception:
        # Fallback: simple linear scaling
        ref_dur  = ref_onsets[-1]  - ref_onsets[0]
        perf_dur = perf_onsets[-1] - perf_onsets[0]
        if perf_dur > 0 and ref_dur > 0:
            scale = ref_dur / perf_dur
            offset = ref_onsets[0] - perf_onsets[0] * scale
            return [Note(n.pitch, n.start*scale+offset, n.end*scale+offset,
                         n.velocity, n.frequency) for n in perf_notes]
        return perf_notes


# ── Tempo analysis ─────────────────────────────────────────────────────────────

def _analyse_tempo(ref_notes: List[Note], perf_notes: List[Note],
                   ref_bpm: float = 120.0) -> TempoAnalysis:
    ref_dur  = ref_notes[-1].end  - ref_notes[0].start  if ref_notes  else 1.0
    perf_dur = perf_notes[-1].end - perf_notes[0].start if perf_notes else 1.0
    ratio    = perf_dur / max(ref_dur, 1e-6)

    perf_bpm = ref_bpm / ratio

    # Local tempo: split into 4 sections, compute ratio per section
    local_ratios = []
    n = len(ref_notes)
    if n >= 8:
        chunk = n // 4
        for i in range(4):
            r_chunk = ref_notes[i*chunk:(i+1)*chunk]
            # find matching perf notes in the same time window
            t0, t1 = r_chunk[0].start - 0.3, r_chunk[-1].end + 0.3
            p_chunk = [n for n in perf_notes if t0 <= n.start <= t1]
            if len(r_chunk) >= 2 and len(p_chunk) >= 2:
                rd = r_chunk[-1].start - r_chunk[0].start
                pd = p_chunk[-1].start - p_chunk[0].start
                local_ratios.append(pd / max(rd, 1e-6))
    if not local_ratios:
        local_ratios = [ratio]

    spread = max(local_ratios) - min(local_ratios)
    if spread > 0.25:
        verdict = "uneven tempo"
    elif ratio < 0.88:
        verdict = "too fast"
    elif ratio > 1.15:
        verdict = "too slow"
    else:
        verdict = "on tempo"

    return TempoAnalysis(
        overall_ratio=ratio,
        overall_bpm_ref=ref_bpm,
        overall_bpm_perf=round(perf_bpm, 1),
        local_ratios=local_ratios,
        verdict=verdict,
    )


# ── Dynamics analysis ──────────────────────────────────────────────────────────

def _analyse_dynamics(matches: List[NoteMatch]) -> DynamicsAnalysis:
    ratios = [m.velocity_ratio for m in matches
              if m.status not in ("missing", "extra") and m.velocity_ratio > 0]
    if not ratios:
        return DynamicsAnalysis(1.0, 0.0, [], [], "no data")

    mean_r = float(np.mean(ratios))
    std_r  = float(np.std(ratios))

    # Find contiguous soft/loud passages (simplified)
    soft_sections  = []
    loud_sections  = []

    if mean_r < 0.55:
        verdict = "too soft overall"
    elif mean_r > 1.55:
        verdict = "too loud overall"
    elif std_r > 0.5:
        verdict = "uneven dynamics"
    else:
        verdict = "good"

    return DynamicsAnalysis(
        mean_velocity_ratio=mean_r,
        std_velocity_ratio=std_r,
        soft_sections=soft_sections,
        loud_sections=loud_sections,
        verdict=verdict,
    )


# ── Per-note greedy matching ───────────────────────────────────────────────────

# Intervals that are purely acoustic artifacts (octave harmonics / sub-harmonics).
# A detected note at exactly these intervals from a ref note at the same time
# is almost certainly a harmonic, not a real wrong note.
HARMONIC_OCTAVE_INTERVALS = {12, 24, 36}  # octave multiples up or down

def _is_harmonic_of(perf_pitch: int, ref_pitch: int) -> bool:
    """True if perf_pitch is an octave harmonic or sub-harmonic of ref_pitch."""
    return abs(perf_pitch - ref_pitch) in HARMONIC_OCTAVE_INTERVALS


def _greedy_match(ref_notes: List[Note], perf_notes: List[Note]) -> List[NoteMatch]:
    """
    Match ref notes to perf notes greedily.

    Matching priority (highest to lowest):
      1. Exact pitch match within onset tolerance  → correct / early / late
      2. Pitch within ±1 semitone (tuning)        → correct / early / late
      3. Octave harmonic at same time              → suppress as extra (not wrong_pitch)
      4. Wrong pitch at same time                  → wrong_pitch
      5. Nothing found                             → missing

    Unmatched perf notes that are octave harmonics of any ref note
    nearby in time are suppressed silently (not counted as extras).
    """
    used = set()
    matches = []
    correct_tol  = CORRECT_ONSET_MS   / 1000.0
    extended_tol = LATE_EARLY_ONSET_MS / 1000.0

    # Build a quick lookup: for each ref note, what pitches are harmonic artifacts?
    def ref_harmonics_near(t: float) -> set:
        """Return set of pitches that are octave harmonics of any ref note near time t."""
        harmonic_pitches = set()
        for rn in ref_notes:
            if abs(rn.start - t) <= extended_tol:
                for interval in HARMONIC_OCTAVE_INTERVALS:
                    harmonic_pitches.add(rn.pitch + interval)
                    harmonic_pitches.add(rn.pitch - interval)
        return harmonic_pitches

    for rn in ref_notes:
        best_idx, best_score = None, float('inf')

        for pi, pn in enumerate(perf_notes):
            if pi in used:
                continue
            od = pn.start - rn.start
            pd = abs(pn.pitch - rn.pitch)
            if abs(od) <= extended_tol and pd <= PITCH_SEMITONES:
                score = abs(od) + pd * 0.1
                if score < best_score:
                    best_score, best_idx = score, pi

        if best_idx is not None:
            pn = perf_notes[best_idx]
            used.add(best_idx)
            od_ms     = (pn.start - rn.start) * 1000
            pd_cents  = (pn.pitch - rn.pitch) * 100.0
            vel_ratio = pn.velocity / max(rn.velocity, 1)

            if abs(od_ms) <= CORRECT_ONSET_MS:
                status = "correct"
            elif od_ms < 0:
                status = "early"
            else:
                status = "late"

            matches.append(NoteMatch(rn, pn, status, od_ms, pd_cents, vel_ratio))
        else:
            # Check for wrong-pitched note at this time
            for pi, pn in enumerate(perf_notes):
                if pi in used:
                    continue
                od = abs(pn.start - rn.start)
                if od <= extended_tol:
                    used.add(pi)
                    od_ms     = (pn.start - rn.start) * 1000
                    pd_cents  = (pn.pitch - rn.pitch) * 100.0
                    vel_ratio = pn.velocity / max(rn.velocity, 1)
                    # Same pitch class but different octave = transcription artifact,
                    # not a real wrong note. Count as correct.
                    if _is_harmonic_of(pn.pitch, rn.pitch):
                        if abs(od_ms) <= CORRECT_ONSET_MS:
                            status = "correct"
                        elif od_ms < 0:
                            status = "early"
                        else:
                            status = "late"
                    else:
                        status = "wrong_pitch"
                    matches.append(NoteMatch(rn, pn, status, od_ms, pd_cents, vel_ratio))
                    break
            else:
                matches.append(NoteMatch(rn, None, "missing", 0, 0, 0))

    # Tag unmatched perf notes: suppress harmonic artifacts, keep genuine extras
    for pi, pn in enumerate(perf_notes):
        if pi not in used:
            harmonics = ref_harmonics_near(pn.start)
            if pn.pitch not in harmonics:
                matches.append(NoteMatch(None, pn, "extra", 0, 0, 0))
            # else: silently drop — it's a harmonic artifact, not a real extra note

    return matches


# ── Per pitch-class accuracy ───────────────────────────────────────────────────

def _per_pitch_accuracy(matches: List[NoteMatch]) -> Dict[str, float]:
    names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    correct = {n: 0 for n in names}
    total   = {n: 0 for n in names}
    for m in matches:
        rn = m.ref_note
        if rn is None:
            continue
        pc = names[rn.pitch % 12]
        total[pc] += 1
        if m.status in ("correct", "early", "late"):
            correct[pc] += 1
    return {pc: round(correct[pc] / total[pc] * 100)
            for pc in names if total[pc] > 0}


# ── mir_eval F1 ────────────────────────────────────────────────────────────────

def _mir_f1(ref_notes: List[Note], perf_notes: List[Note]) -> Tuple[float, float, float]:
    if not ref_notes or not perf_notes:
        return 0.0, 0.0, 0.0
    ref_iv  = np.array([[n.start, n.start + 0.5] for n in ref_notes])
    perf_iv = np.array([[n.start, n.start + 0.5] for n in perf_notes])
    ref_hz  = np.array([n.frequency for n in ref_notes])
    perf_hz = np.array([n.frequency for n in perf_notes])
    try:
        p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_iv, ref_hz, perf_iv, perf_hz,
            onset_tolerance=MIR_ONSET_TOL,
            pitch_tolerance=MIR_PITCH_TOL,
            offset_ratio=None,
        )
        return float(p), float(r), float(f)
    except Exception:
        return 0.0, 0.0, 0.0


# ── Main entry point ───────────────────────────────────────────────────────────

def compare_notes(
    ref_notes:          List[Note],
    perf_notes:         List[Note],
    ref_bpm:            float = 120.0,
    dtw_align:          bool  = True,
    amplitude_threshold: float = 0.0,   # 0 = no filter; set 0.55 for synth, 0 for acoustic
) -> ComparisonResult:
    """
    Full comparison pipeline:
      1. Sort + clip perf to ref window
      2. Optional amplitude filter (for synth audio only)
      3. DTW tempo alignment
      4. mir_eval F1 score
      5. Per-note greedy matching with early/late/wrong_pitch detail
      6. Tempo + dynamics analysis
    """
    if not ref_notes or not perf_notes:
        empty_tempo = TempoAnalysis(1.0, ref_bpm, ref_bpm, [1.0], "no data")
        empty_dyn   = DynamicsAnalysis(1.0, 0.0, [], [], "no data")
        return ComparisonResult(
            matches=[], accuracy=0, precision=0, recall=0, f1=0,
            n_correct=0, n_early=0, n_late=0, n_missing=len(ref_notes),
            n_wrong_pitch=0, n_extra=0,
            mean_onset_dev_ms=0, std_onset_dev_ms=0, mean_pitch_dev_cents=0,
            tempo=empty_tempo, dynamics=empty_dyn, per_pitch={},
        )

    # 1. Sort
    ref_notes  = sorted(ref_notes,  key=lambda n: (n.start, n.pitch))
    perf_notes = sorted(perf_notes, key=lambda n: (n.start, n.pitch))

    # 2. Clip perf to ref time window (±0.5s buffer)
    t0, t1 = ref_notes[0].start - 0.5, ref_notes[-1].end + 0.5
    perf_notes = [n for n in perf_notes if t0 <= n.start <= t1]

    # 3. Optional amplitude filter (only use for synthesized audio)
    if amplitude_threshold > 0:
        perf_notes = [n for n in perf_notes if n.velocity >= amplitude_threshold * 127]

    # 4. Tempo analysis (before alignment, so we capture the actual tempo ratio)
    tempo = _analyse_tempo(ref_notes, perf_notes, ref_bpm)
    print(f"  Tempo: {tempo.overall_bpm_perf:.1f} BPM  ({tempo.verdict})")

    # 5. DTW alignment
    if dtw_align and len(perf_notes) >= 4:
        perf_notes = _dtw_align(ref_notes, perf_notes)

    # 6. mir_eval F1
    precision, recall, f1 = _mir_f1(ref_notes, perf_notes)
    print(f"  F1: {f1*100:.1f}%  P: {precision*100:.1f}%  R: {recall*100:.1f}%")

    # 7. Per-note matching
    matches = _greedy_match(ref_notes, perf_notes)

    n_correct     = sum(1 for m in matches if m.status == "correct")
    n_early       = sum(1 for m in matches if m.status == "early")
    n_late        = sum(1 for m in matches if m.status == "late")
    n_missing     = sum(1 for m in matches if m.status == "missing")
    n_wrong_pitch = sum(1 for m in matches if m.status == "wrong_pitch")
    n_extra       = sum(1 for m in matches if m.status == "extra")

    timed_devs = [m.onset_dev_ms for m in matches
                  if m.status in ("correct", "early", "late") and m.perf_note]
    pitch_devs = [m.pitch_dev_cents for m in matches if m.perf_note]

    # 8. Dynamics
    dynamics = _analyse_dynamics(matches)

    return ComparisonResult(
        matches=matches,
        accuracy=f1, precision=precision, recall=recall, f1=f1,
        n_correct=n_correct, n_early=n_early, n_late=n_late,
        n_missing=n_missing, n_wrong_pitch=n_wrong_pitch, n_extra=n_extra,
        mean_onset_dev_ms =float(np.mean(timed_devs)) if timed_devs else 0.0,
        std_onset_dev_ms  =float(np.std(timed_devs))  if timed_devs else 0.0,
        mean_pitch_dev_cents=float(np.mean(pitch_devs)) if pitch_devs else 0.0,
        tempo=tempo,
        dynamics=dynamics,
        per_pitch=_per_pitch_accuracy(matches),
    )