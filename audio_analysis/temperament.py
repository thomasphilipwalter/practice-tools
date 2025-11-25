import numpy as np

from audio_analysis.f0 import (
    get_f0_contour,
    segment_notes,
    find_nearest_12tet_note,
    frequency_to_cents,
)

JI_RATIOS = {
    "major":          [1/1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2/1],
    "natural_minor":  [1/1, 9/8, 6/5, 4/3, 3/2, 8/5,  9/5, 2/1],
    "harmonic_minor": [1/1, 9/8, 6/5, 4/3, 3/2, 8/5, 15/8, 2/1],
    "melodic_minor":  [1/1, 9/8, 6/5, 4/3, 3/2, 5/3, 15/8, 2/1], # ascending
}

# Semitone offsets (mod 12) for degrees 1 -> 7 of each scale, relative to tonic
SCALE_SEMITONES = {
    "major":          [0, 2, 4, 5, 7, 9, 11],
    "natural_minor":  [0, 2, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "melodic_minor":  [0, 2, 3, 5, 7, 9, 11],  # ascending
}

NOTE_TO_SEMITONE = {
    "C": 0,  "C#": 1,  "Db": 1,
    "D": 2,  "D#": 3,  "Eb": 3,
    "E": 4,
    "F": 5,  "F#": 6,  "Gb": 6,
    "G": 7,  "G#": 8,  "Ab": 8,
    "A": 9,  "A#": 10, "Bb": 10,
    "B": 11,
}

# -----------------------------
# Note / pitch class helpers
# -------------------------------

def parse_note_name(note: str):
    """
    Parse something like 'C4', 'F#3', 'Bb5' into (pitch_class, octave).
    pitch_class: 'C', 'C#', 'Db', ...
    octave: int
    """
    note = note.strip()
    # Octave is last char (or last 2), librosa uses ex: C#4
    if len(note) >= 3 and note[-2].isdigit():
        pitch = note[:-2]
        octave = int(note[-2:])
    else:
        pitch = note[:-1]
        octave = int(note[-1])
    return pitch, octave


def note_to_midi(note: str) -> int:
    """
    Convert a note like 'C4', 'F#3', 'Bb5' to MIDI (A4=69).
    """
    pitch, octave = parse_note_name(note)
    if pitch not in NOTE_TO_SEMITONE:
        raise ValueError(f"Unknown pitch name: {pitch}")
    semitone = NOTE_TO_SEMITONE[pitch]
    # C-1 = 0 -> C0 = 12, C4 = 60, A4 = 69
    return (octave + 1) * 12 + semitone

def pitchclass_to_semitone(pitch: str) -> int:
    if pitch not in NOTE_TO_SEMITONE:
        raise ValueError(f"Unknown pitch: {pitch}")
    return NOTE_TO_SEMITONE[pitch]

# -----------------------------
# Tuning helpers
# -----------------------------

def get_root_freq_based_on_a(tuning_freq: float, tonic_with_octave: str) -> float:
    """
    Given A4 tuning frequency (ex 440.0) and a tonic like 'E4', 'F#3',
    return the equal tempered frequency of that tonic under this tuning.

    Uses:
        f = A4 * 2^((midi - 69) / 12)
    """
    midi_root = note_to_midi(tonic_with_octave)
    semitone_diff = midi_root - 69  # distance from A4
    return tuning_freq * (2.0 ** (semitone_diff / 12.0))


# ---------------------------------
# Scale degree mapping
# ------------------------------

def get_scale_degree_and_octave_offset(note_name: str, tonic_pc: str, mode: str, tonic_octave: int) -> tuple:
    """
    Return (degree, octave_offset) for note_name relative to the tonic.
    degree is 1 -> 7, octave_offset counts how many octaves above the tonic.
    """
    if mode not in SCALE_SEMITONES:
        raise ValueError(f"Unsupported mode: {mode}")

    scale_semitones = SCALE_SEMITONES[mode]

    pitch, octave = parse_note_name(note_name)
    note_midi = note_to_midi(note_name)
    tonic_pc = tonic_pc.strip()

    if tonic_pc not in NOTE_TO_SEMITONE:
        raise ValueError(f"Unknown tonic pitch class: {tonic_pc}")

    tonic_midi_base = note_to_midi(f"{tonic_pc}{tonic_octave}")

    semitone_diff = note_midi - tonic_midi_base
    if semitone_diff < 0:
        # For descending phrases, allow negative diff by adding 12s until non-negative
        octave_shift = int(np.floor(semitone_diff / 12))
        semitone_diff -= 12 * octave_shift
    octave_offset, diff_mod_12 = divmod(semitone_diff, 12)

    if diff_mod_12 not in scale_semitones:
        return None, None

    degree_index = scale_semitones.index(diff_mod_12)
    degree = degree_index + 1

    return degree, octave_offset

# ----------------------------
# Just Intonation target freq
# ----------------------------

def get_ji_target_frequency(
    tuning_freq: float,
    tonic_with_octave: str,
    mode: str,
    degree: int,
    octave_offset: int = 0,
) -> float:
    """
    Compute JI target frequency for a given scale degree and octave offset.

    - tuning_freq: A4 tuning (e.g. 440.0)
    - tonic_with_octave: e.g. 'E4' (used to derive base tonic frequency)
    - mode: 'major', 'natural_minor', 'harmonic_minor', 'melodic_minor'
    - degree: 1 -> 8 
    - octave_offset: how many octaves above the tonic this specific note is
    """
    if mode not in JI_RATIOS:
        raise ValueError(f"Unsupported mode: {mode}")
    ratios = JI_RATIOS[mode]

    if degree < 1 or degree > len(ratios):
        raise ValueError(f"Degree {degree} out of range for mode {mode}")

    f_root = get_root_freq_based_on_a(tuning_freq, tonic_with_octave)
    ratio = ratios[degree - 1]
    return f_root * ratio * (2 ** octave_offset)


# ---------------------------
# Main analysis functions
# ---------------------------

def analyze_temperament(
    f0,
    voiced_flag,
    voiced_probs,
    sr,
    hop_length=512,
    tuning_freq=440.0,
    tonic_pc="E",        # e.g. 'E', 'F#', 'Bb'
    tonic_octave=4,      # e.g. 4 for 'E4' as the scale tonic
    mode="major",
):
    """
    Analyze just-intonation intonation for a simple scale in a given key.

    Assumptions:
    - Audio is a clean scale (ascending or descending) in a single key.
    - Key is specified by tonic pitch class + octave + mode.
    - A4 tuning (tuning_freq) is known.
    - f0, voiced_flag, voiced_probs come from f0.get_f0_contour.

    Returns:
    - dict with per-note results and summary stats.
    """
    f0 = np.asarray(f0)
    voiced_flag = np.asarray(voiced_flag).astype(bool)

    # Segment into note events
    note_segments = segment_notes(f0, voiced_flag)

    tonic_note_full = f"{tonic_pc}{tonic_octave}"

    per_note_results = []
    all_cents_off = []

    for (start, end) in note_segments:
        segment_f0 = f0[start:end]
        segment_f0 = segment_f0[~np.isnan(segment_f0)]
        if len(segment_f0) == 0:
            continue

        measured_freq = float(np.median(segment_f0))

        # Nearest ET note given this tuning
        note_name, nearest_freq, _dev_cents = find_nearest_12tet_note(
            measured_freq, tuning_freq=tuning_freq
        )
        if note_name is None:
            continue

        # Map this note into scale degree + octave offset
        degree, octave_offset = get_scale_degree_and_octave_offset(
            note_name, tonic_pc, mode, tonic_octave
        )
        if degree is None:
            # Not a diatonic scale degree; skip (shouldn't happen with clean scale input)
            continue

        # For display, treat the top tonic (octave) as degree 8, if it appears as 1 + octave_offset
        display_degree = degree
        if degree == 1 and octave_offset > 0:
            display_degree = 8

        ji_target = get_ji_target_frequency(
            tuning_freq=tuning_freq,
            tonic_with_octave=tonic_note_full,
            mode=mode,
            degree=degree,
            octave_offset=octave_offset,
        )

        # Cents difference from the JI target
        cents_off = 1200 * np.log2(measured_freq / ji_target)
        all_cents_off.append(cents_off)

        per_note_results.append({
            "start_frame": int(start),
            "end_frame": int(end),
            "measured_freq_hz": measured_freq,
            "nearest_12tet_note": note_name,
            "scale_degree": display_degree,  # 1..7 or 8
            "raw_degree": degree,            # 1..7
            "octave_offset": octave_offset,
            "ji_target_freq_hz": ji_target,
            "cents_off_from_ji": cents_off,
        })

    summary = {
        "tuning_a4_hz": tuning_freq,
        "key": f"{tonic_pc} {mode}",
        "tonic_note": tonic_note_full,
        "n_notes": len(per_note_results),
        "mean_abs_cents_off": float(np.mean(np.abs(all_cents_off))) if all_cents_off else None,
        "per_note": per_note_results,
    }

    return summary

