import librosa
import numpy as np

from audio_analysis.f0 import segment_notes, find_nearest_12tet_note

def analyze_pitch(f0, voiced_flag, voiced_probs, sr, hop_length=512, tuning_freq=440.0, min_note_duration_frames=10, silence_threshold_frames=5, pitch_change_threshold_semitones=0.5, pitch_average_window=5):
    """
    Use segment_notes note boundaries to calculate average pitch across notes.
    Return data structure containing note metadata for each note.
    """
    
    # Calculate note boundaries
    notes = segment_notes(
        f0, 
        voiced_flag,
        min_note_duration_frames=min_note_duration_frames,
        silence_threshold_frames=silence_threshold_frames,
        pitch_change_threshold_semitones=pitch_change_threshold_semitones,
        pitch_average_window=pitch_average_window
    )

    # Build time array: Map frames to seconds using SAME hop_length as librosa.pyin
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    
    note_analyses = []
    
    for start_frame, end_frame in notes:
        # Extract f0 and voiced flag for the given time frame
        note_f0 = f0[start_frame:end_frame]
        note_voiced = voiced_flag[start_frame:end_frame]
        
        # Filter out unvoiced or nan f0
        valid_f0 = note_f0[note_voiced & ~np.isnan(note_f0)]
        
        if len(valid_f0) == 0:
            continue
        
        # Average F0 for this note
        avg_f0 = np.mean(valid_f0)
        
        # Find nearest 12-TET note and deviation
        note_name, nearest_freq, deviation_cents = find_nearest_12tet_note(avg_f0, tuning_freq)
        
        if note_name is None:
            continue
        
        # Calculate note duration
        start_time = times[start_frame]
        end_time = librosa.frames_to_time([end_frame], sr=sr, hop_length=hop_length)[0]
        duration = end_time - start_time
        
        note_analyses.append({
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'avg_f0': avg_f0,
            'nearest_note': note_name,
            'nearest_freq': nearest_freq,
            'deviation_cents': deviation_cents,
            'start_frame': start_frame,
            'end_frame': end_frame
        })
    
    return note_analyses