import os
from typing import Any, Dict, List

from audio_analysis.f0 import get_f0_contour, analyze_notes

def run_note_analysis(audio_path: str) -> Dict[str, Any]:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    f0, voiced_flag, voiced_probs, sr = get_f0_contour(audio_path)
    note_analyses = analyze_notes(f0, voiced_flag, voiced_probs, sr)

    return {
        "note_count": len(note_analyses),
        "notes": [_normalize_note(note) for note in note_analyses],
    }


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