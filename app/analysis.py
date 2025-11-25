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

    return {
        "times": [None if np.isnan(x) else float(x) for x in vibrato_results["times"]],
        "rate_hz": [None if np.isnan(x) else float(x) for x in vibrato_results["rate_hz"]],
        "width_cents": [None if np.isnan(x) else float(x) for x in vibrato_results["width_cents"]],
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
    
    result = analyze_temperament(
        f0, voiced_flag, voiced_probs, sr,
        tuning_freq=tuning_freq,
        tonic_pc=tonic_pc,
        tonic_octave=tonic_octave,
        mode=mode,
        hop_length=hop_length
    )
    
    # Normalize for JSON serialization: replace NaN with None, ensure all floats are native Python types
    def clean_value(v):
        """Convert numpy types and NaN to JSON-safe values."""
        if isinstance(v, (np.integer, np.floating)):
            if np.isnan(v):
                return None
            return float(v) if isinstance(v, np.floating) else int(v)
        if isinstance(v, float) and np.isnan(v):
            return None
        return v
    
    # Clean the summary
    cleaned_summary = {
        "tuning_a4_hz": clean_value(result["tuning_a4_hz"]),
        "key": result["key"],
        "tonic_note": result["tonic_note"],
        "n_notes": int(result["n_notes"]),
        "mean_abs_cents_off": clean_value(result["mean_abs_cents_off"]),
        "per_note": [
            {
                "start_frame": int(note["start_frame"]),
                "end_frame": int(note["end_frame"]),
                "measured_freq_hz": clean_value(note["measured_freq_hz"]),
                "nearest_12tet_note": note["nearest_12tet_note"],
                "scale_degree": int(note["scale_degree"]),
                "raw_degree": int(note["raw_degree"]),
                "octave_offset": int(note["octave_offset"]),
                "ji_target_freq_hz": clean_value(note["ji_target_freq_hz"]),
                "cents_off_from_ji": clean_value(note["cents_off_from_ji"]),
            }
            for note in result["per_note"]
        ]
    }
    
    return cleaned_summary

def _normalize_note(note: Dict[str, Any]) -> Dict[str, Any]:
    def safe_float(v):
        """Convert to float, replacing NaN with None."""
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            return None
        return float(v) if v is not None else None
    
    return {
        "start_time": safe_float(note["start_time"]),
        "end_time": safe_float(note["end_time"]),
        "duration": safe_float(note["duration"]),
        "avg_f0": safe_float(note["avg_f0"]),
        "nearest_note": str(note["nearest_note"]),
        "nearest_freq": safe_float(note["nearest_freq"]),
        "deviation_cents": safe_float(note["deviation_cents"]),
        "start_frame": int(note["start_frame"]),
        "end_frame": int(note["end_frame"]),
    }