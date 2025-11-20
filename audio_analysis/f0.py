import librosa
import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Import an audio file and process F0 contour"
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the audio file",
    )
    return parser.parse_args()

def get_f0_contour(audio_path):
    y, sr = librosa.load(audio_path)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, 
        sr=sr, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'))
    return f0, voiced_flag, voiced_probs, sr

def segment_notes(f0, voiced_flag, min_note_duration_frames=10, silence_threshold_frames=5, pitch_change_threshold_semitones=0.5, pitch_average_window=5):
    """
    Segment F0 contour into individual notes based on voicing and pitch changes.
    
    Parameters: 
    - min_note_duration_frames: Minimum frames for a note to be considered valid
    - silence_threshold_frames: Frames of silence to consider as note boundary
    - pitch_change_threshold_semitones: Minimum pitch change (in semitones) to consider as new note
    - pitch_average_window: Number of recent frames to average for reference pitch
    
    Returns:
    - List of tuples: (start_frame, end_frame) for each note segment
    """

    notes = []
    in_note = False
    note_start = 0
    recent_pitches = []  # Track recent pitches for running average
    
    for i in range(len(voiced_flag)):
        if voiced_flag[i] and not in_note:
            # Start of new note
            in_note = True
            note_start = i
            current_pitch = f0[i]
            if not np.isnan(current_pitch):
                recent_pitches = [current_pitch]
        elif voiced_flag[i] and in_note:
            # We're in a note and still voiced - check for pitch change
            current_pitch = f0[i]
            
            if not np.isnan(current_pitch):
                # Calculate average of recent pitches as reference
                if len(recent_pitches) > 0:
                    reference_pitch = np.mean(recent_pitches)
                    
                    # Calculate pitch change in semitones
                    pitch_change_semitones = 12 * np.log2(current_pitch / reference_pitch)
                    
                    # If pitch change is significant, end current note and start new one
                    if abs(pitch_change_semitones) >= pitch_change_threshold_semitones:
                        note_end = i
                        if note_end - note_start >= min_note_duration_frames:
                            notes.append((note_start, note_end))
                        # Start new note
                        note_start = i
                        recent_pitches = [current_pitch]
                    else:
                        # Update recent pitches (sliding window)
                        recent_pitches.append(current_pitch)
                        if len(recent_pitches) > pitch_average_window:
                            recent_pitches.pop(0)
                else:
                    # First valid pitch in note
                    recent_pitches.append(current_pitch)
        elif not voiced_flag[i] and in_note:
            # Check if this is a sustained silence (note end)
            silence_count = 0
            j = i
            while j < len(voiced_flag) and not voiced_flag[j]:
                silence_count += 1
                j += 1
                if silence_count >= silence_threshold_frames:
                    break
            
            if silence_count >= silence_threshold_frames:
                # End of note
                note_end = i
                if note_end - note_start >= min_note_duration_frames:
                    notes.append((note_start, note_end))
                in_note = False
                recent_pitches = []
    
    # Handle note that extends to end of audio
    if in_note and len(voiced_flag) - note_start >= min_note_duration_frames:
        notes.append((note_start, len(voiced_flag)))
    
    return notes

def frequency_to_cents(freq, reference_freq=440.0):
    """
    Convert frequency to cents relative to a reference frequency.
    """
    if freq <= 0 or np.isnan(freq):
        return np.nan
    return 1200 * np.log2(freq / reference_freq)

def cents_to_frequency(cents, reference_freq=440.0):
    """
    Convert cents to frequency.
    """
    return reference_freq * (2 ** (cents / 1200))

def find_nearest_12tet_note(freq):
    """
    Find the nearest 12-TET note to a given frequency. 

    Returns:
    - (note_name, note_freq, deviation_cents)
    """
    if freq <= 0 or np.isnan(freq):
        return None, None, None
    
    midi_num = librosa.hz_to_midi(freq)     # frequency to MIDI (continuous)
    nearest_midi = int(round(midi_num))     # Round to nearest int (12-TET MIDI note)
    nearest_freq = librosa.midi_to_hz(nearest_midi)    # Convert back to frequency
    note_name = librosa.midi_to_note(nearest_midi)      # Get note name
    deviation_cents = frequency_to_cents(freq, nearest_freq)    # Calculate deviation in cents

    return note_name, nearest_freq, deviation_cents


def analyze_notes(f0, voiced_flag, voiced_probs, sr, hop_length=512):
    """
    Analyze F0 contour to extract note segments and calculate deviations from 12-TET
    
    Returns:
    - list of dictionaries with note info
    """
    notes = segment_notes(f0, voiced_flag)
    hop_length = 512
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    
    note_analyses = []
    
    for start_frame, end_frame in notes:
        # Extract F0 values for this note (only voiced frames)
        note_f0 = f0[start_frame:end_frame]
        note_voiced = voiced_flag[start_frame:end_frame]
        
        # Filter to only valid (voiced) F0 values
        valid_f0 = note_f0[note_voiced & ~np.isnan(note_f0)]
        
        if len(valid_f0) == 0:
            continue
        
        # Average F0 for this note
        avg_f0 = np.mean(valid_f0)
        
        # Find nearest 12-TET note and deviation
        note_name, nearest_freq, deviation_cents = find_nearest_12tet_note(avg_f0)
        
        if note_name is None:
            continue
        
        # Calculate note duration - handle edge case where end_frame might be out of bounds
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

def plot_analysis(f0, voiced_flag, voiced_probs, sr, note_analyses):
    """
    Plot F0 contour with note segments and deviation information.
    """
    hop_length = 512
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Plot 1: F0 contour with note segments
    axes[0].plot(times, f0, linewidth=1, color='lightblue', alpha=0.5, label='F0 contour')
    
    # Highlight note segments
    for note in note_analyses:
        start_idx = note['start_frame']
        end_idx = note['end_frame']
        note_times = times[start_idx:end_idx]
        note_f0 = f0[start_idx:end_idx]
        
        # Plot note segment
        axes[0].plot(note_times, note_f0, linewidth=2, color='blue')
        
        # Mark average pitch
        avg_time = (note['start_time'] + note['end_time']) / 2
        axes[0].plot(avg_time, note['avg_f0'], 'ro', markersize=8)
        
        # Mark nearest 12-TET pitch
        axes[0].axhline(y=note['nearest_freq'], xmin=note['start_time']/times[-1], 
                        xmax=note['end_time']/times[-1], 
                        color='red', linestyle='--', alpha=0.5)
        
        # Add note label
        axes[0].text(avg_time, note['avg_f0'], 
                    f"{note['nearest_note']}\n{note['deviation_cents']:.1f}¢",
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_title('F0 Contour with Note Segments and 12-TET Deviations')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_yscale('log')  # Log scale for better visualization
    
    # Plot 2: Deviation in cents for each note
    if note_analyses:
        note_indices = np.arange(len(note_analyses))
        deviations = [note['deviation_cents'] for note in note_analyses]
        note_names = [note['nearest_note'] for note in note_analyses]
        
        axes[1].bar(note_indices, deviations, color='steelblue', alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
        axes[1].set_ylabel('Deviation from 12-TET (cents)')
        axes[1].set_xlabel('Note Number')
        axes[1].set_title('Pitch Deviation from Nearest 12-TET Note')
        axes[1].set_xticks(note_indices)
        axes[1].set_xticklabels(note_names, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()


def print_analysis(note_analyses):
    """
    Print a summary table of note analyses.
    """
    print("\n" + "="*80)
    print("Note Analysis: Deviation from 12-TET")
    print("="*80)
    print(f"{'Note':<8} {'Avg F0 (Hz)':<12} {'12-TET Note':<12} {'12-TET F0 (Hz)':<15} {'Deviation (cents)':<18}")
    print("-"*80)
    
    for i, note in enumerate(note_analyses):
        print(f"{i+1:<8} {note['avg_f0']:<12.2f} {note['nearest_note']:<12} "
              f"{note['nearest_freq']:<15.2f} {note['deviation_cents']:<18.2f}")
    
    if note_analyses:
        avg_deviation = np.mean([abs(note['deviation_cents']) for note in note_analyses])
        print("-"*80)
        print(f"Average absolute deviation: {avg_deviation:.2f} cents")
    print("="*80 + "\n")


def main():
    # Get path
    args = parse_args()
    audio_path = args.audio_path

    # Check path exists
    if not os.path.exists(audio_path):
        print(f"Error: file does not exist: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # Check if file
    if not os.path.isfile(audio_path):
        print(f"Error: path is not a file: {audio_path}", file=sys.stderr)
        sys.exit(1)

    # Extract F0 contour
    f0, voiced_flag, voiced_probs, sr = get_f0_contour(audio_path)
    
    # Analyze notes
    note_analyses = analyze_notes(f0, voiced_flag, voiced_probs, sr)
    
    # Print results
    print_analysis(note_analyses)
    
    # Plot results
    plot_analysis(f0, voiced_flag, voiced_probs, sr, note_analyses)

if __name__ == "__main__":
    main()