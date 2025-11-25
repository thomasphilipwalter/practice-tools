import librosa
import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.ndimage import median_filter
from scipy.signal import medfilt

def interpolate_f0_within_voiced_blocks(f0, voiced_flag):
    """
    Linearly interpolate NaN values in the F0 contour, within contiguous voiced regions.
    Unvoiced regions left unchanged (preserve note boundaries/rests)
    Returns interpolated f0 contour
    """

    # Input validation
    f0 = np.asarray(f0, dtype=float)
    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    if f0.shape != voiced_flag.shape:
        raise ValueError("f0 and voiced_flag must have same shape (interpolation func)")
    
    f0_interp = f0.copy()

    # Get indices of all unvoiced frames
    voiced_indices = np.flatnonzero(voiced_flag)
    if len(voiced_indices) == 0:
        return f0_interp
    
    # Find boundaries between contiguous voiced blocks
    gaps = np.where(np.diff(voiced_indices) > 1)[0]
    block_starts = np.concatenate(([0], gaps + 1))
    block_ends = np.concatenate((gaps, [len(voiced_indices) - 1]))

    # Iterate over each voiced block
    for start_idx, end_idx in zip(block_starts, block_ends):

        # Extract block data
        block_indices = voiced_indices[start_idx:end_idx + 1]
        block_f0 = f0_interp[block_indices]

        # Inside this block, may still have NaNs
        not_nan = ~np.isnan(block_f0)
        
        # If all nans, or none nans, continue
        if not np.any(not_nan) or np.all(not_nan):
            continue

        # 1d linear interpolation over time indices
        known_x = block_indices[not_nan]
        known_y = block_f0[not_nan]
        missing_x = block_indices[~not_nan]

        interp_vals = np.interp(missing_x, known_x, known_y)

        # Write interpolated values back
        f0_interp[missing_x] = interp_vals
    
    return f0_interp

def smooth_f0_contour(f0, voiced_flag, voiced_probs, median_filter_size=5, moving_average_window=3, min_voiced_prob=0.1):
    """
    Smooth F0 contour to remove noise and outliers
    Return regular contour data, but smoothed
    """

    # Convert all input to numpy arrays
    f0 = np.asarray(f0, dtype=float)
    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    voiced_probs = np.asarray(voiced_probs, dtype=float)

    # Ensure 1-1 mapping between arrays (should never fail)
    if not (f0.shape == voiced_flag.shape == voiced_probs.shape):
        raise ValueError("f0, voiced_flag, and voiced_probs must have the same shape")

    # Apply minimum probability threshold
    smoothed_voiced_flag = voiced_flag & (voiced_probs >= min_voiced_prob) # Elementwise set true ONLY if both
    smoothed_voiced_probs = voiced_probs.copy()
    smoothed_voiced_probs[~smoothed_voiced_flag] = 0.0 # If not voiced, set prob to 0

    # Set f0 to NaN if not voiced
    smoothed_f0 = f0.copy()
    smoothed_f0[~smoothed_voiced_flag] = np.nan 

    # Interpolate NaNs WITHIN each voiced block
    smoothed_f0 = interpolate_f0_within_voiced_blocks(smoothed_f0, smoothed_voiced_flag)
    
    # If no voiced frames remain, return early
    if not np.any(smoothed_voiced_flag):
        return smoothed_f0, smoothed_voiced_flag, smoothed_voiced_probs

    # Helper to smooth a 1D block with median + moving average
    def _smooth_block(block_values):
        x = block_values.copy()

        # Median filter
        if median_filter_size > 1 and median_filter_size % 2 == 1:
            x = medfilt(x, kernel_size=median_filter_size)

        # Moving average
        if moving_average_window > 1:
            kernel = np.ones(moving_average_window) / moving_average_window
            pad = moving_average_window // 2
            padded = np.pad(x, (pad, pad), mode="edge")
            x = np.convolve(padded, kernel, mode="valid")

        return x

    # Apply smoothing block-wise inside voiced segments
    voiced_indices = np.flatnonzero(smoothed_voiced_flag) 
    gaps = np.where(np.diff(voiced_indices) > 1)[0]
    block_starts = np.concatenate(([0], gaps + 1)) # First index of each gap block
    block_ends   = np.concatenate((gaps, [len(voiced_indices) - 1])) # Last index of each gap block

    for start_idx, end_idx in zip(block_starts, block_ends):
        block_indices = voiced_indices[start_idx:end_idx + 1]
        block_f0 = smoothed_f0[block_indices]

        # Defensive check, shouldn't execute
        if block_f0.size == 0: 
            continue
        
        # Apply smoothing
        smoothed_block = _smooth_block(block_f0)
        smoothed_f0[block_indices] = smoothed_block

    return smoothed_f0, smoothed_voiced_flag, smoothed_voiced_probs

def get_f0_contour(audio_path, smooth=False, hop_length=512, median_filter_size=5, moving_average_window=3, min_voiced_prob=0.1):
    """
    Extract f0 contour from an audio file, optionally smooth.
    Return fundamental frequencies, voicing, probability voiced, sample rate, hop length.
    """
    y, sr = librosa.load(audio_path) # Load audio (returns audio time-series samples, sample rate)
    f0, voiced_flag, voiced_probs = librosa.pyin( # Get contour using probablistic YIN algorithm
        y, 
        sr=sr, 
        fmin=librosa.note_to_hz('C2'), 
        fmax=librosa.note_to_hz('C7'),
        hop_length=hop_length)

    if smooth:
        f0, voiced_flag, voiced_probs = smooth_f0_contour(
            f0, 
            voiced_flag, 
            voiced_probs, 
            median_filter_size=median_filter_size,
            moving_average_window=moving_average_window,
            min_voiced_prob=min_voiced_prob
        )
    
    return f0, voiced_flag, voiced_probs, sr, hop_length

def segment_notes(f0, voiced_flag, min_note_duration_frames=10, silence_threshold_frames=5, pitch_change_threshold_semitones=0.5, pitch_average_window=5):
    """
    Segment f0 into individual notes based on voicing and pitch changes
    NOTE: Does some redundant work if smoothing was applied on f0 contour

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
        # New voicing —> start new note
        if voiced_flag[i] and not in_note:
            in_note = True
            note_start = i
            current_pitch = f0[i]
            if not np.isnan(current_pitch):
                recent_pitches = [current_pitch]
        # Not new voicing -> Check if pitch changed
        elif voiced_flag[i] and in_note:
            current_pitch = f0[i]
            # Frequency detected -> Check if pitch change
            if not np.isnan(current_pitch):
                # Recent pitches exist -> Caclulate average & compare
                if len(recent_pitches) > 0:
                    reference_pitch = np.mean(recent_pitches)
                    # Calculate pitch change in semitones
                    pitch_change_semitones = 12 * np.log2(current_pitch / reference_pitch)
                    # If above threshold, start new note
                    if abs(pitch_change_semitones) >= pitch_change_threshold_semitones:
                        note_end = i
                        if note_end - note_start >= min_note_duration_frames:
                            notes.append((note_start, note_end))
                        # Start new note
                        note_start = i
                        recent_pitches = [current_pitch]
                    # Not above threshold, keep note and add to recent pitches
                    else:
                        recent_pitches.append(current_pitch)
                        if len(recent_pitches) > pitch_average_window:
                            recent_pitches.pop(0)
                # Recent pitches don't exist -> firsrt valid pitch
                else:
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

def find_nearest_12tet_note(freq, tuning_freq=440.0):
    """
    Find the nearest 12-tone equal temperament note to a given frequency.
    
    Parameters:
    - freq: Frequency in Hz
    - tuning_freq: A4 tuning frequency (default 440 Hz)
    
    Returns:
    - (note_name, note_freq, deviation_cents)
    """
    if freq <= 0 or np.isnan(freq):
        return None, None, None
    
    # Calculate MIDI note number using the specific tuning frequency
    midi_num = 69 + 12 * np.log2(freq / tuning_freq)
    nearest_midi = int(round(midi_num))

    # Convert MIDI back to frequency using the specified tuning
    nearest_freq = tuning_freq * (2 ** ((nearest_midi - 69) / 12))
    note_name = librosa.midi_to_note(nearest_midi)
    deviation_cents = frequency_to_cents(freq, nearest_freq)

    return note_name, nearest_freq, deviation_cents