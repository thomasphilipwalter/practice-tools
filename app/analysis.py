import os
from typing import Any, Dict, List
import numpy as np

from audio_analysis.pitch import analyze_pitch
from audio_analysis.vibrato import analyze_vibrato
from audio_analysis.temperament import analyze_temperament

def run_pitch_analysis(f0: np.ndarray, voiced_flag: np.ndarray, voiced_probs: np.ndarray, 
                      sr: int, tuning_freq: float = 440.0) -> Dict[str, Any]:
    """
    Run pitch/intonation analysis on F0 contour data.
    
    Parameters:
    - f0: F0 contour array
    - voiced_flag: Boolean array indicating voiced frames
    - voiced_probs: Probability array for voicing
    - sr: Sample rate
    - tuning_freq: A4 tuning frequency (default 440 Hz)
    """
    note_analyses = analyze_pitch(f0, voiced_flag, voiced_probs, sr, tuning_freq=tuning_freq)

    return {
        "note_count": len(note_analyses),
        "notes": [_normalize_note(note) for note in note_analyses],
    }


def run_vibrato_analysis(f0: np.ndarray, voiced_flag: np.ndarray, voiced_probs: np.ndarray,
                         sr: int, tuning_freq: float = 440.0) -> Dict[str, Any]:
    """
    Run vibrato analysis on F0 contour data.
    
    Parameters:
    - f0: F0 contour array
    - voiced_flag: Boolean array indicating voiced frames
    - voiced_probs: Probability array for voicing
    - sr: Sample rate
    - tuning_freq: A4 tuning frequency (default 440 Hz)
    """
    # TODO: IMPLEMENT
    return {}

def run_temperament_analysis(f0: np.ndarray, voiced_flag: np.ndarray, voiced_probs: np.ndarray,
                             sr: int, tuning_freq: float = 440.0, 
                             starting_note: Optional[str] = None,
                             mode: Optional[str] = None) -> Dict[str, Any]:
    """
    Run temperament analysis on F0 contour data.
    
    Parameters:
    - f0: F0 contour array
    - voiced_flag: Boolean array indicating voiced frames
    - voiced_probs: Probability array for voicing
    - sr: Sample rate
    - tuning_freq: A4 tuning frequency (default 440 Hz)
    - starting_note: Starting note of the scale (e.g., "E4", "F#3")
    - mode: Scale mode ("major", "natural_minor", "harmonic_minor", "melodic_minor")
    """
    if not starting_note or not mode:
        raise ValueError("starting_note and mode are required for temperament analysis")
    
    # Parse starting_note into pitch class and octave
    from audio_analysis.temperament import parse_note_name
    tonic_pc, tonic_octave = parse_note(starting_note)
    
    return analyze_temperament(
        f0, voiced_flag, voiced_probs, sr,
        tuning_freq=tuning_freq,
        tonic_pc=tonic_pc,
        tonic_octave=tonic_octave,
        mode=mode
    )

def _normalize_note(note: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "start_time": float(note["start_time"]),
        "end_time": float(note["end_time"]),
        "duration": float(note["duration"]),
        "avg_f0": float(note["avg_f0"]),
        "nearest_note": str(note["nearest_note"]),
        "nearest_freq": float(note["nearest_freq"]),
        "deviation_cents": float(note["deviation_cents"]),
        "start_frame": int(note["start_frame"]),
        "end_frame": int(note["end_frame"]),
    }