import os
from typing import Any, Dict, List, Optional
import numpy as np

from audio_analysis.pitch import analyze_pitch
from audio_analysis.vibrato import analyze_vibrato
from audio_analysis.temperament import analyze_temperament
from audio_analysis.f0 import smooth_f0_contour

def run_pitch_analysis(f0: np.ndarray, voiced_flag: np.ndarray, voiced_probs: np.ndarray, 
                      sr: int, tuning_freq: float = 440.0, hop_length: int = 512) -> Dict[str, Any]:
    """
    Run pitch/intonation analysis on F0 contour data.
    
    Parameters:
    - f0: F0 contour array
    - voiced_flag: Boolean array indicating voiced frames
    - voiced_probs: Probability array for voicing
    - sr: Sample rate
    - tuning_freq: A4 tuning frequency (default 440 Hz)
    """
    note_analyses = analyze_pitch(f0, voiced_flag, voiced_probs, sr, tuning_freq=tuning_freq, hop_length=hop_length)

    return {
        "note_count": len(note_analyses),
        "notes": [_normalize_note(note) for note in note_analyses],
    }


def run_vibrato_analysis(f0: np.ndarray, voiced_flag: np.ndarray, voiced_probs: np.ndarray,
                         sr: int, tuning_freq: float = 440.0, hop_length: int = 512) -> Dict[str, Any]:
    """
    Run vibrato analysis on F0 contour data.
    """

    # Smooth contour for vib (performs much better)
    f0_smooth, voiced_flag_smooth, voiced_probs_smooth = smooth_f0_contour(
                f0,
                voiced_flag,
                voiced_probs, 
    )

    # Perform vib analysis
    vibrato_results = analyze_vibrato(
                f0_smooth, 
                voiced_flag_smooth, 
                voiced_probs_smooth, 
                sr, 
                tuning_freq=tuning_freq,
                window_seconds=0.8,
                step_seconds=0.1,
                min_voiced_fraction=0.7,
                hop_length=hop_length
    )

    return vibrato_results
    return {
        "times": vibrato_results["times"].tolist(),
        "rate_hz": vibrato_results["rate_hz"].tolist(),
        "width_cents": vibrato_results["width_cents"].tolist(),
    }


def run_temperament_analysis(f0: np.ndarray, voiced_flag: np.ndarray, voiced_probs: np.ndarray,
                             sr: int, tuning_freq: float = 440.0, 
                             starting_note: Optional[str] = None,
                             mode: Optional[str] = None,
                             hop_length: int = 512) -> Dict[str, Any]:
    """
    Run temperament analysis on F0 contour data.
    """
    if not starting_note or not mode:
        raise ValueError("starting_note and mode are required for temperament analysis")
    
    # Parse starting_note into pitch class and octave
    from audio_analysis.temperament import parse_note_name
    tonic_pc, tonic_octave = parse_note_name(starting_note)
    
    return analyze_temperament(
        f0, voiced_flag, voiced_probs, sr,
        tuning_freq=tuning_freq,
        tonic_pc=tonic_pc,
        tonic_octave=tonic_octave,
        mode=mode,
        hop_length=hop_length
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