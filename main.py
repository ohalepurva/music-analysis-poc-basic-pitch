"""
main.py — AI Music Teacher
──────────────────────────
Compare a MIDI reference against an audio performance (or another MIDI).
Uses the ByteDance piano transcription model for audio analysis.

Usage:
  # MIDI vs Audio
  python main.py --midi sheet.mid --audio recording.wav

  # MIDI vs MIDI
  python main.py --midi ref.mid --midi2 performance.mid

  # Save visual outputs to a folder
  python main.py --midi sheet.mid --audio recording.wav --output-dir ./results

  # Built-in self-test (no files needed)
  python main.py --self-test
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

from piano import (
    transcribe_audio,
    midi_to_notes,
    compare_notes,
    draw_piano_roll,
    draw_timing_deviation_chart,
    draw_accuracy_by_pitch_class,
)


def run_comparison(ref_notes, perf_notes, output_dir=None, title="Piano Comparison"):
    print(f"\n{'─'*60}")
    print(f"  Reference notes  : {len(ref_notes)}")
    print(f"  Performance notes: {len(perf_notes)}")
    print(f"{'─'*60}")

    t0 = time.time()
    result = compare_notes(ref_notes, perf_notes, dtw_align=True)
    print(f"  Compared in {time.time() - t0:.2f}s\n")
    print(result.summary())

    # Per-pitch-class breakdown
    if result.per_pitch:
        print("\n  Per-pitch-class accuracy:")
        for pc, acc in sorted(result.per_pitch.items(), key=lambda x: -x[1]):
            bar = "█" * int(acc / 5)
            print(f"    {pc:3s}  {bar:<20}  {acc:5.1f}%")

    # Save outputs
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        draw_piano_roll(result, os.path.join(output_dir, "piano_roll.png"), title=title)
        draw_timing_deviation_chart(result, os.path.join(output_dir, "timing_deviation.png"))
        draw_accuracy_by_pitch_class(result, os.path.join(output_dir, "pitch_class_accuracy.png"))

        report = {
            "accuracy_%":               round(result.accuracy * 100, 2),
            "precision_%":              round(result.precision * 100, 2),
            "recall_%":                 round(result.recall * 100, 2),
            "f1_%":                     round(result.f1 * 100, 2),
            "correct":                  result.n_correct,
            "missing":                  result.n_missing,
            "extra":                    result.n_extra,
            "wrong_pitch":              result.n_wrong_pitch,
            "mean_onset_deviation_ms":  round(result.mean_onset_dev_ms, 2),
            "std_onset_deviation_ms":   round(result.std_onset_dev_ms, 2),
            "mean_pitch_deviation_cents": round(result.mean_pitch_dev_cents, 2),
            "per_pitch_accuracy_%": {
                k: round(v, 1) for k, v in result.per_pitch.items()
            },
            "notes": [
                {
                    "status":           m.status,
                    "ref_pitch":        m.ref_note.pitch  if m.ref_note  else None,
                    "perf_pitch":       m.perf_note.pitch if m.perf_note else None,
                    "ref_start_s":      round(m.ref_note.start,  3) if m.ref_note  else None,
                    "perf_start_s":     round(m.perf_note.start, 3) if m.perf_note else None,
                    "onset_dev_ms":     round(m.onset_dev_ms, 1),
                    "pitch_dev_cents":  round(m.pitch_dev_cents, 1),
                }
                for m in result.matches
            ],
        }
        with open(os.path.join(output_dir, "report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Results saved → {output_dir}/")

    return result


def self_test(output_dir="./output/self_test"):
    """Verify the full pipeline works without needing real files."""
    import pretty_midi
    import numpy as np
    from piano.audio_to_notes import Note

    print("\n" + "="*60)
    print("  SELF-TEST: MIDI vs MIDI identical (expect 100%)")
    print("="*60)

    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano = pretty_midi.Instrument(program=0)
    t = 0.0
    for pitch in [60, 62, 64, 65, 67, 69, 71, 72]:   # C major scale
        piano.notes.append(pretty_midi.Note(80, pitch, t, t + 0.45))
        t += 0.5
    for chord in [(60,64,67), (62,65,69), (64,67,71)]:  # chords
        for p in chord:
            piano.notes.append(pretty_midi.Note(90, p, t, t + 0.9))
        t += 1.0
    pm.instruments.append(piano)

    os.makedirs(output_dir, exist_ok=True)
    midi_path = os.path.join(output_dir, "test.mid")
    pm.write(midi_path)

    ref   = midi_to_notes(midi_path)
    perf  = midi_to_notes(midi_path)
    result = run_comparison(ref, perf, output_dir=os.path.join(output_dir, "identical"),
                            title="Self-Test: Identical")
    assert result.f1 >= 0.999, f"Expected 100%, got {result.f1*100:.1f}%"
    print(f"\n  ✓ Passed: {result.f1*100:.1f}%")

    print("\n" + "="*60)
    print("  SELF-TEST 2: ±20ms timing jitter (expect ~95-100%)")
    print("="*60)
    rng = np.random.default_rng(42)
    jittered = [
        Note(n.pitch, max(0, n.start + rng.uniform(-0.02, 0.02)),
             max(0.03, n.end + rng.uniform(-0.02, 0.02)),
             n.velocity, n.frequency)
        for n in ref
    ]
    result2 = run_comparison(ref, jittered, output_dir=os.path.join(output_dir, "jittered"),
                             title="Self-Test: ±20ms Jitter")
    print(f"\n  ✓ Completed: {result2.f1*100:.1f}%")
    print(f"\n  All outputs → {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="AI Music Teacher — Piano performance analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--midi",       help="Reference MIDI file (.mid)")
    parser.add_argument("--audio",      help="Performance audio file (.wav/.mp3/.flac)")
    parser.add_argument("--midi2",      help="Performance MIDI file (instead of --audio)")
    parser.add_argument("--output-dir", default="./output", help="Where to save results")
    parser.add_argument("--no-visuals", action="store_true", help="Skip chart generation")
    parser.add_argument("--self-test",  action="store_true", help="Run built-in test")

    args = parser.parse_args()

    if args.self_test:
        self_test(output_dir=args.output_dir)
        return

    if not args.midi:
        parser.print_help()
        sys.exit(1)

    ref_notes = midi_to_notes(args.midi)
    if not ref_notes:
        print("ERROR: No notes found in reference MIDI.")
        sys.exit(1)

    if args.audio:
        perf_notes = transcribe_audio(args.audio)
        title = f"{Path(args.midi).stem} vs {Path(args.audio).stem}"
    elif args.midi2:
        perf_notes = midi_to_notes(args.midi2)
        title = f"{Path(args.midi).stem} vs {Path(args.midi2).stem}"
    else:
        print("ERROR: Provide --audio or --midi2.")
        parser.print_help()
        sys.exit(1)

    if not perf_notes:
        print("ERROR: No notes found in performance input.")
        sys.exit(1)

    run_comparison(
        ref_notes, perf_notes,
        output_dir=None if args.no_visuals else args.output_dir,
        title=title,
    )


if __name__ == "__main__":
    main()
