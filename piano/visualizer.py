"""
visualizer.py
─────────────
Generates a visual piano roll diff between reference and performance.
Color coding:
  - Green  : correct note (matched within tolerance)
  - Red    : missing note (in reference, not in performance)
  - Orange : extra note  (in performance, not in reference)
  - Purple : wrong pitch  (timing matched but wrong note)
  - Blue   : wrong timing (pitch matched but too early/late)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from typing import List, Optional
from .note_model import Note, PIANO_MIDI_MIN, PIANO_MIDI_MAX
from .comparator import ComparisonResult, NoteMatch


# ─── Color scheme ─────────────────────────────────────────────────────────────
STATUS_COLORS = {
    "correct":     "#2ECC71",   # green
    "early":       "#64B5F6",   # blue
    "late":        "#F39C12",   # orange
    "missing":     "#E74C3C",   # red
    "extra":       "#888899",   # grey
    "wrong_pitch": "#9B59B6",   # purple
}

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
WHITE_KEYS = {0, 2, 4, 5, 7, 9, 11}  # C D E F G A B


def midi_to_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def draw_piano_roll(
    result: ComparisonResult,
    output_path: str,
    title: str = "Piano Roll Comparison",
    max_duration: Optional[float] = None,
    show_piano_keys: bool = True,
):
    """
    Draw an annotated piano roll diff and save to output_path.
    """
    # Collect all notes for axis scaling
    all_notes_with_time = []
    for m in result.matches:
        if m.ref_note:
            all_notes_with_time.append(m.ref_note)
        if m.perf_note:
            all_notes_with_time.append(m.perf_note)

    if not all_notes_with_time:
        print("  Nothing to visualize.")
        return

    t_max = max(n.end for n in all_notes_with_time) + 0.5
    if max_duration:
        t_max = min(t_max, max_duration)

    pitches_present = [n.pitch for n in all_notes_with_time]
    pitch_min = max(PIANO_MIDI_MIN, min(pitches_present) - 3)
    pitch_max = min(PIANO_MIDI_MAX, max(pitches_present) + 3)
    pitch_range = pitch_max - pitch_min + 1

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig_width = min(28, max(14, t_max * 1.5))
    fig_height = max(6, pitch_range * 0.22)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    fig.patch.set_facecolor('#1A1A2E')
    ax.set_facecolor('#16213E')

    # ── Piano key background ──────────────────────────────────────────────────
    for midi in range(pitch_min, pitch_max + 1):
        pc = midi % 12
        if pc not in WHITE_KEYS:  # black key row
            ax.axhspan(midi - 0.5, midi + 0.5, color='#0F3460', alpha=0.4, zorder=0)
        else:
            ax.axhspan(midi - 0.5, midi + 0.5, color='#1A1A2E', alpha=0.2, zorder=0)

    # ── Grid lines (every octave C) ───────────────────────────────────────────
    for midi in range(pitch_min, pitch_max + 1):
        if midi % 12 == 0:  # C
            ax.axhline(y=midi, color='#FFFFFF', alpha=0.15, linewidth=0.8, zorder=1)

    # Vertical grid (every beat-ish, at 0.5s intervals)
    for t in np.arange(0, t_max, 0.5):
        ax.axvline(x=t, color='#FFFFFF', alpha=0.08, linewidth=0.5, zorder=1)
    for t in np.arange(0, t_max, 1.0):
        ax.axvline(x=t, color='#FFFFFF', alpha=0.15, linewidth=0.8, zorder=1)

    # ── Draw notes ────────────────────────────────────────────────────────────
    NOTE_HEIGHT = 0.75
    BORDER_WIDTH = 1.5

    def draw_note(ax, note: Note, color: str, alpha: float = 0.85, zorder: int = 3):
        if note.end > t_max:
            return
        x = note.start
        y = note.pitch - NOTE_HEIGHT / 2
        w = max(note.duration, 0.02)
        h = NOTE_HEIGHT
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.01",
            linewidth=BORDER_WIDTH,
            edgecolor='white',
            facecolor=color,
            alpha=alpha,
            zorder=zorder,
        )
        ax.add_patch(rect)

    for m in result.matches:
        color = STATUS_COLORS.get(m.status, "#AAAAAA")

        if m.ref_note and m.status in ("missing", "wrong_pitch"):
            # Draw reference as semi-transparent
            draw_note(ax, m.ref_note, color, alpha=0.45, zorder=2)

        if m.perf_note:
            draw_note(ax, m.perf_note, color, alpha=0.9, zorder=3)
        elif m.ref_note:
            draw_note(ax, m.ref_note, color, alpha=0.7, zorder=3)

    # ── Y-axis labels (note names) ────────────────────────────────────────────
    y_ticks = []
    y_labels = []
    for midi in range(pitch_min, pitch_max + 1):
        if midi % 12 in (0, 4, 7, 9):  # C E G A — less cluttered
            y_ticks.append(midi)
            y_labels.append(midi_to_name(midi))

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8, color='#AAAAAA')
    ax.set_ylim(pitch_min - 1, pitch_max + 1)

    # ── X-axis ────────────────────────────────────────────────────────────────
    ax.set_xlim(-0.1, t_max)
    ax.set_xlabel("Time (seconds)", color='#AAAAAA', fontsize=10)
    ax.tick_params(colors='#AAAAAA')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

    # ── Title & accuracy watermark ────────────────────────────────────────────
    accuracy_pct = result.accuracy * 100
    ax.set_title(
        f"{title}  |  Accuracy: {accuracy_pct:.1f}%  |  "
        f"✓ {result.n_correct} correct  ◀ {result.n_early} early  ▶ {result.n_late} late  "
        f"✗ {result.n_missing} missing  ✱ {result.n_wrong_pitch} wrong  + {result.n_extra} extra",
        color='white', fontsize=12, pad=12,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=STATUS_COLORS["correct"],     label=f"Correct ({result.n_correct})"),
        mpatches.Patch(color=STATUS_COLORS["early"],       label=f"Early ({result.n_early})"),
        mpatches.Patch(color=STATUS_COLORS["late"],        label=f"Late ({result.n_late})"),
        mpatches.Patch(color=STATUS_COLORS["missing"],     label=f"Missing ({result.n_missing})"),
        mpatches.Patch(color=STATUS_COLORS["extra"],       label=f"Extra ({result.n_extra})"),
        mpatches.Patch(color=STATUS_COLORS["wrong_pitch"], label=f"Wrong pitch ({result.n_wrong_pitch})"),
    ]
    ax.legend(
        handles=legend_patches,
        loc='upper right',
        framealpha=0.3,
        facecolor='#1A1A2E',
        edgecolor='#555577',
        labelcolor='white',
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Piano roll saved → {output_path}")


def draw_timing_deviation_chart(
    result: ComparisonResult,
    output_path: str,
):
    """
    Scatter plot: onset time vs. timing deviation for each matched note.
    Shows systematic drift or random jitter in the performance.
    """
    matched = [m for m in result.matches if m.status not in ("missing", "extra") and m.ref_note and m.perf_note]
    if not matched:
        return

    times  = [m.ref_note.start for m in matched]
    devs   = [m.onset_dev_ms for m in matched]
    colors = [STATUS_COLORS.get(m.status, "#AAAAAA") for m in matched]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    fig.patch.set_facecolor('#1A1A2E')

    # ── Top: timing deviation over time ──────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor('#16213E')
    ax.scatter(times, devs, c=colors, alpha=0.7, s=30, zorder=3)
    ax.axhline(0, color='white', linewidth=1.0, alpha=0.5)
    ax.axhspan(-50, 50, color='#2ECC71', alpha=0.08)  # ±50ms tolerance band
    ax.set_ylabel("Onset deviation (ms)", color='#AAAAAA')
    ax.set_title("Timing Deviation per Note", color='white', fontsize=11)
    ax.tick_params(colors='#AAAAAA')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

    # Rolling mean (trend line)
    if len(times) > 5:
        from scipy.ndimage import uniform_filter1d
        sorted_idx = np.argsort(times)
        t_sorted = np.array(times)[sorted_idx]
        d_sorted = np.array(devs)[sorted_idx]
        window = max(3, len(d_sorted) // 10)
        trend = uniform_filter1d(d_sorted, size=window)
        ax.plot(t_sorted, trend, color='#F39C12', linewidth=2.0,
                alpha=0.8, label='Trend', zorder=4)
        ax.legend(labelcolor='white', facecolor='#1A1A2E',
                  edgecolor='#555577', framealpha=0.3)

    # ── Bottom: pitch deviation histogram ─────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor('#16213E')
    pitch_devs = [m.pitch_dev_cents for m in matched if abs(m.pitch_dev_cents) < 200]
    if pitch_devs:
        n, bins, patches = ax2.hist(pitch_devs, bins=40, color='#3498DB',
                                     alpha=0.8, edgecolor='#1A1A2E')
        ax2.axvline(0, color='white', linewidth=1.5, alpha=0.7)
        ax2.axvline(np.mean(pitch_devs), color='#F39C12', linewidth=1.5,
                    linestyle='--', label=f'Mean: {np.mean(pitch_devs):.1f}¢')
        ax2.set_xlabel("Pitch deviation (cents)", color='#AAAAAA')
        ax2.set_ylabel("Count", color='#AAAAAA')
        ax2.set_title("Pitch Deviation Distribution", color='white', fontsize=11)
        ax2.tick_params(colors='#AAAAAA')
        ax2.legend(labelcolor='white', facecolor='#1A1A2E',
                   edgecolor='#555577', framealpha=0.3)
        for spine in ax2.spines.values():
            spine.set_edgecolor('#333355')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Deviation chart saved → {output_path}")


def draw_accuracy_by_pitch_class(
    result: ComparisonResult,
    output_path: str,
):
    """Bar chart showing accuracy per pitch class (C, C#, D, ...)."""
    note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    accuracies = [result.per_pitch.get(n, 0) for n in note_names]
    colors = ['#2ECC71' if a >= 90 else '#F39C12' if a >= 70 else '#E74C3C'
              for a in accuracies]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#1A1A2E')
    ax.set_facecolor('#16213E')

    bars = ax.bar(note_names, accuracies, color=colors, edgecolor='#1A1A2E', linewidth=1.2)
    ax.axhline(100, color='#2ECC71', linewidth=1.0, alpha=0.4, linestyle='--')
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", color='#AAAAAA')
    ax.set_title("Note Accuracy by Pitch Class", color='white', fontsize=12)
    ax.tick_params(colors='#AAAAAA')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333355')

    # Value labels on bars
    for bar, val in zip(bars, accuracies):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.0f}%', ha='center', va='bottom',
                    color='white', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Pitch class chart saved → {output_path}")