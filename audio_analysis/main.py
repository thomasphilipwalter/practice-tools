#!/usr/bin/env python3
"""
CLI tool for testing audio analysis features independently.
Usage:
    python -m audio_analysis.main <audio_path> [--analysis-type pitch|vibrato|temperament|all] 
        [--tuning-freq 440] [--starting-note E4] [--mode major] [--plot]
"""

import argparse
import os
import sys
from typing import Optional

import librosa
import matplotlib.pyplot as plt
import numpy as np

from audio_analysis.f0 import get_f0_contour, segment_notes
from audio_analysis.pitch import analyze_pitch
from audio_analysis.vibrato import analyze_vibrato, aggregate_vibrato_per_note
from audio_analysis.temperament import analyze_temperament

def time_to_frames(time_seconds, sr, hop_length):
    """Convert time duration to frame count."""
    frame_rate = sr / float(hop_length)
    return max(1, int(round(time_seconds * frame_rate)))

def print_pitch_results(note_analyses: list, tuning_freq: float):
    """Print pitch analysis results in a formatted table."""
    print("\n" + "="*80)
    print(f"Pitch Analysis Results (Tuning: A4 = {tuning_freq} Hz)")
    print("="*80)
    print(f"{'Note':<8} {'Start (s)':<12} {'End (s)':<12} {'Duration (s)':<15} "
          f"{'Avg F0 (Hz)':<12} {'12-TET Note':<12} {'Deviation (cents)':<18}")
    print("-"*80)
    
    for i, note in enumerate(note_analyses):
        print(f"{i+1:<8} {note['start_time']:<12.3f} {note['end_time']:<12.3f} "
              f"{note['duration']:<15.3f} {note['avg_f0']:<12.2f} "
              f"{note['nearest_note']:<12} {note['deviation_cents']:<18.2f}")
    
    if note_analyses:
        avg_deviation = np.mean([abs(note['deviation_cents']) for note in note_analyses])
        print("-"*80)
        print(f"Total notes detected: {len(note_analyses)}")
        print(f"Average absolute deviation: {avg_deviation:.2f} cents")
    print("="*80 + "\n")

def print_vibrato_results(results: dict):
    """Print vibrato analysis results."""
    print("\n" + "="*80)
    print("Vibrato Analysis Results")
    print("="*80)
    if not results:
        print("Vibrato analysis not yet implemented.")
    else:
        # TODO: Format vibrato results when implemented
        print(results)
    print("="*80 + "\n")

def print_temperament_results(results: dict):
    """Print temperament analysis results."""
    print("\n" + "="*80)
    print("Temperament Analysis Results")
    print("="*80)
    
    if not results:
        print("Temperament analysis not yet implemented.")
    else:
        print(f"Key: {results.get('key', 'N/A')}")
        print(f"Tonic: {results.get('tonic_note', 'N/A')}")
        print(f"Tuning: A4 = {results.get('tuning_a4_hz', 'N/A')} Hz")
        print(f"Number of notes analyzed: {results.get('n_notes', 0)}")
        print(f"Mean absolute deviation from JI: {results.get('mean_abs_cents_off', 0):.2f} cents")
        print("-"*80)
        
        per_note = results.get('per_note', [])
        if per_note:
            print(f"{'Degree':<10} {'12-TET Note':<12} {'Measured (Hz)':<15} {'JI Target (Hz)':<18} {'Deviation (cents)':<18}")
            print("-"*80)
            
            for note in per_note:
                degree = note.get('scale_degree', 'N/A')
                note_name = note.get('nearest_12tet_note', 'N/A')
                measured = note.get('measured_freq_hz', 0)
                ji_target = note.get('ji_target_freq_hz', 0)
                deviation = note.get('cents_off_from_ji', 0)
                
                print(f"{degree:<10} {note_name:<12} {measured:<15.2f} {ji_target:<18.2f} {deviation:<18.2f}")
    
    print("="*80 + "\n")

def plot_pitch_analysis(f0, voiced_flag, voiced_probs, sr, note_analyses, hop_length, save_path=None):
    """
    Plot F0 contour with note segments and deviation information.
    Shows two separate plots sequentially, or saves them if save_path is provided.
    
    Parameters
    ----------
    save_path : str, optional
        Base path to save plots. If provided, saves two files:
        - {save_path}_f0_contour.png
        - {save_path}_deviations.png
        If None, displays plots interactively.
    """
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    
    # Plot 1: F0 contour with note segments
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 6))
    
    ax1.plot(times, f0, linewidth=1, color='lightblue', alpha=0.5, label='F0 contour')
    
    # Highlight note segments
    for note in note_analyses:
        start_idx = note['start_frame']
        end_idx = note['end_frame']
        note_times = times[start_idx:end_idx]
        note_f0 = f0[start_idx:end_idx]
        
        # Plot note segment
        ax1.plot(note_times, note_f0, linewidth=2, color='blue')
        
        # Mark average pitch
        avg_time = (note['start_time'] + note['end_time']) / 2
        ax1.plot(avg_time, note['avg_f0'], 'ro', markersize=8)
        
        # Add note label
        ax1.text(avg_time, note['avg_f0'], 
                f"{note['nearest_note']}\n{note['deviation_cents']:.1f}¢",
                ha='center', va='bottom', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_xlabel('Time (s)')
    ax1.set_title('F0 Contour with Note Segments')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_yscale('log')
    
    plt.tight_layout()
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(f"{save_path}_f0_contour.png", dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print(f"  Saved: {save_path}_f0_contour.png")
    else:
        plt.show()  # Show first plot
    
    # Plot 2: Deviation in cents for each note
    if note_analyses:
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
        
        note_indices = np.arange(len(note_analyses))
        deviations = [note['deviation_cents'] for note in note_analyses]
        note_names = [note['nearest_note'] for note in note_analyses]
        
        ax2.bar(note_indices, deviations, color='steelblue', alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_ylabel('Deviation from 12-TET (cents)')
        ax2.set_xlabel('Note Number')
        ax2.set_title('Pitch Deviation from Nearest 12-TET Note')
        ax2.set_xticks(note_indices)
        ax2.set_xticklabels(note_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}_deviations.png", dpi=150, bbox_inches='tight')
            plt.close(fig2)
            print(f"  Saved: {save_path}_deviations.png")
        else:
            plt.show()  # Show second plot
    
def plot_vibrato_analysis(vibrato_results: dict):
    """Visualize vibrato rate and width over time."""
    if not vibrato_results:
        print("No vibrato results to plot.")
        return

    times = vibrato_results.get("times")
    rate_hz = vibrato_results.get("rate_hz")
    width_cents = vibrato_results.get("width_cents")

    if times is None or rate_hz is None or width_cents is None:
        print("Vibrato results missing required fields; cannot plot.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    # Vibrato rate
    axes[0].plot(times, rate_hz, color="mediumslateblue", linewidth=1.5, label="Rate (Hz)")
    axes[0].scatter(times, rate_hz, color="mediumslateblue", s=12, alpha=0.6)
    axes[0].set_ylabel("Rate (Hz)")
    axes[0].set_title("Vibrato Rate Over Time")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Vibrato width
    axes[1].plot(times, width_cents, color="darkorange", linewidth=1.5, label="Width (cents)")
    axes[1].scatter(times, width_cents, color="darkorange", s=12, alpha=0.6)
    axes[1].set_ylabel("Width (cents)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Vibrato Width Over Time")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def print_per_note_vibrato_results(per_note_vibrato: list):
    """Pretty-print vibrato stats aggregated per detected note segment."""
    if not per_note_vibrato:
        print("\nNo per-note vibrato data available.\n")
        return

    print("\n" + "=" * 80)
    print("Per-Note Vibrato Summary")
    print("=" * 80)
    header = f"{'Note #':<8} {'Start (s)':<12} {'End (s)':<12} {'Rate (Hz)':<12} {'Width (cents)':<15}"
    print(header)
    print("-" * len(header))

    for note in per_note_vibrato:
        print(f"{note['note_index']:<8} "
              f"{note['start_time']:<12.3f} "
              f"{note['end_time']:<12.3f} "
              f"{note['vibrato_rate_hz'] if not np.isnan(note['vibrato_rate_hz']) else '—':<12} "
              f"{note['vibrato_width_cents'] if not np.isnan(note['vibrato_width_cents']) else '—':<15}")

    print("=" * 80 + "\n")

def plot_per_note_vibrato(per_note_vibrato: list):
    """Bar charts of vibrato rate/width per note segment."""
    if not per_note_vibrato:
        print("No per-note vibrato data to plot.")
        return

    note_indices = [note["note_index"] for note in per_note_vibrato]
    rates = [note["vibrato_rate_hz"] for note in per_note_vibrato]
    widths = [note["vibrato_width_cents"] for note in per_note_vibrato]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].bar(note_indices, rates, color="mediumpurple", alpha=0.8)
    axes[0].set_ylabel("Rate (Hz)")
    axes[0].set_title("Per-Note Vibrato Rate")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(note_indices, widths, color="darkorange", alpha=0.8)
    axes[1].set_ylabel("Width (cents)")
    axes[1].set_xlabel("Note Index")
    axes[1].set_title("Per-Note Vibrato Width")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_f0_contour(f0, voiced_flag, sr, hop_length=512, smoothed_f0=None, title="F0 Contour"):
    """
    Plot F0 contour(s) over time.
    
    Parameters
    ----------
    f0 : array-like
        Original F0 contour (may contain NaNs)
    voiced_flag : array-like
        Boolean array indicating voiced frames
    sr : int
        Sample rate
    hop_length : int
        Hop length used for F0 extraction (default: 512)
    smoothed_f0 : array-like, optional
        Smoothed F0 contour to plot alongside original
    title : str, optional
        Plot title (default: "F0 Contour")
    """
    import matplotlib.pyplot as plt
    import librosa
    import numpy as np
    
    # Convert to numpy arrays
    f0 = np.asarray(f0, dtype=float)
    voiced_flag = np.asarray(voiced_flag, dtype=bool)
    
    # Create time axis
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Plot original F0 contour
    # Only plot voiced frames to avoid cluttering with NaN regions
    voiced_mask = voiced_flag & ~np.isnan(f0)
    if np.any(voiced_mask):
        ax.plot(times[voiced_mask], f0[voiced_mask], 
               linewidth=1.5, color='darkblue', alpha=0.7, 
               label='Original F0', marker='o', markersize=2)
    
    # Plot smoothed F0 if provided
    if smoothed_f0 is not None:
        smoothed_f0 = np.asarray(smoothed_f0, dtype=float)
        smoothed_voiced_mask = voiced_flag & ~np.isnan(smoothed_f0)
        if np.any(smoothed_voiced_mask):
            ax.plot(times[smoothed_voiced_mask], smoothed_f0[smoothed_voiced_mask],
                   linewidth=2, color='orange', alpha=0.9,
                   label='Smoothed F0', marker='s', markersize=2)
    
    # Formatting
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(title)
    ax.set_yscale('log')  # Log scale for better musical frequency visualization
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_temperament_analysis(results: dict, save_path=None):
    """
    Plot temperament analysis results showing deviation from Just Intonation.
    
    Parameters
    ----------
    results : dict
        Results from analyze_temperament with keys:
        - 'per_note': list of dicts with scale_degree, nearest_12tet_note, 
          measured_freq_hz, ji_target_freq_hz, cents_off_from_ji
        - 'key', 'tonic_note', 'tuning_a4_hz', 'mean_abs_cents_off'
    save_path : str, optional
        Base path to save plots. If provided, saves files instead of displaying.
    """
    if not results or not results.get('per_note'):
        print("No temperament data to plot.")
        return
    
    per_note = results['per_note']
    key = results.get('key', 'Unknown')
    mean_abs_dev = results.get('mean_abs_cents_off', 0)
    
    # Extract data
    degrees = [note['scale_degree'] for note in per_note]
    note_names = [note['nearest_12tet_note'] for note in per_note]
    deviations = [note['cents_off_from_ji'] for note in per_note]
    measured_freqs = [note['measured_freq_hz'] for note in per_note]
    ji_targets = [note['ji_target_freq_hz'] for note in per_note]
    
    # Create two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Deviation from JI in cents (bar chart)
    colors = ['steelblue' if d >= 0 else 'coral' for d in deviations]
    bars = ax1.bar(range(len(degrees)), deviations, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax1.set_ylabel('Deviation from JI (cents)', fontsize=12)
    ax1.set_xlabel('Scale Degree', fontsize=12)
    ax1.set_title(f'Temperament Analysis: {key}\nDeviation from Just Intonation (Mean: {mean_abs_dev:.2f} cents)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(degrees)))
    ax1.set_xticklabels([f"{d}\n({n})" for d, n in zip(degrees, note_names)], fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([min(deviations) - 5, max(deviations) + 5])
    
    # Add value labels on bars
    for i, (bar, dev) in enumerate(zip(bars, deviations)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{dev:.1f}',
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # Plot 2: Measured frequencies vs JI targets (scatter/line plot)
    x_pos = range(len(degrees))
    ax2.plot(x_pos, measured_freqs, 'o-', color='steelblue', linewidth=2, 
             markersize=10, label='Measured Frequency', alpha=0.8)
    ax2.plot(x_pos, ji_targets, 's--', color='darkorange', linewidth=2, 
             markersize=8, label='JI Target Frequency', alpha=0.8)
    
    # Add note labels
    for i, (x, freq, name) in enumerate(zip(x_pos, measured_freqs, note_names)):
        ax2.annotate(name, (x, freq), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
    
    ax2.set_ylabel('Frequency (Hz)', fontsize=12)
    ax2.set_xlabel('Scale Degree', fontsize=12)
    ax2.set_title('Measured vs Just Intonation Target Frequencies', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"Degree {d}" for d in degrees], fontsize=10)
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(f"{save_path}_temperament.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: {save_path}_temperament.png")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Test audio analysis features (f0, pitch, vibrato, temperament)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""""""
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the audio file"
    )
    parser.add_argument(
        "--analysis-type",
        type=str,
        choices=["f0", "pitch", "vibrato", "temperament", "all"],
        default="pitch",
        help="Type of analysis to run (default: pitch)"
    )
    parser.add_argument(
        "--tuning-freq",
        type=float,
        default=440.0,
        help="A4 tuning frequency in Hz (default: 440.0)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display plots for analysis results (default: False)"
    )
    parser.add_argument(
        "--starting-note",
        type=str,
        default=None,
        help="Starting note of the scale for temperament analysis (e.g., 'E4', 'F#3', 'Bb5')"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["major", "natural_minor", "harmonic_minor", "melodic_minor"],
        default=None,
        help="Scale mode for temperament analysis (required if --analysis-type includes temperament)"
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=512,
        help="Hop length for f0 extraction"
    )
    parser.add_argument(
        "--smooth",
        action="store_true",
        help="Use smooth scale for analysis"
    )
    parser.add_argument(
        "--save-plots",
        type=str,
        default=None,
        help="Base path to save plots instead of displaying"
    )
    args = parser.parse_args()
    
    # Validate audio file exists
    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found: {args.audio_path}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.isfile(args.audio_path):
        print(f"Error: Path is not a file: {args.audio_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading audio file: {args.audio_path}")
    print(f"Tuning frequency: A4 = {args.tuning_freq} Hz")
    print(f"Analysis type(s): {args.analysis_type}")
    print("-" * 80)
    
    # ------------- EXTRACT F0 CONTOUR ---------------------------
    print("Extracting F0 contour...")
    try:
        # Normal
        f0, voiced_flag, voiced_probs, sr, hop_length = get_f0_contour(args.audio_path, smooth=False, hop_length=args.hop_length)
        print(f"F0 contour extracted: {len(f0)} frames, sample rate: {sr} Hz")
        # Smooth
        f0_smooth, voiced_flag_smooth, voiced_probs_smooth, sr, hop_length = get_f0_contour(args.audio_path, smooth=True, hop_length=args.hop_length)
        print(f"Smooth F0 contour extracted: {len(f0)} frames, sample rate: {sr} Hz")
    except Exception as e:
        print(f"Error extracting F0 contour: {e}", file=sys.stderr)
        sys.exit(1)
    

    # -------------------------------------------------------------
    # -------------- RUN ANALYSIS ----------------------------------
    # -------------------------------------------------------------
    

    # --- F0 ---
    if args.analysis_type == "f0":
        # Plot original F0
        plot_f0_contour(f0, voiced_flag, sr, hop_length)

        # Plot with smoothed
        plot_f0_contour(f0, voiced_flag, sr, hop_length, smoothed_f0=f0_smooth)

    # --- SET SMOOTH ---
    if args.smooth:
        f0 = f0_smooth
        voiced_flag = voiced_flag_smooth
        voiced_probs = voiced_probs_smooth

    # Define time-based parameters (in seconds)
    min_note_duration_sec = 0.08  # Minimum note duration: 50ms
    silence_threshold_sec = 0.04  # Silence threshold: 20ms  
    pitch_average_window_sec = 0.12  # Averaging window: 100ms

    # Convert to frames based on current hop_length
    min_note_duration_frames = time_to_frames(min_note_duration_sec, sr, hop_length)
    silence_threshold_frames = time_to_frames(silence_threshold_sec, sr, hop_length)
    pitch_average_window_frames = time_to_frames(pitch_average_window_sec, sr, hop_length)

    # --- PITCH ---
    if args.analysis_type in ["pitch", "all"]:
        print("\nRunning pitch analysis...")
        try:

            # Get note analysis results
            note_analyses = analyze_pitch(f0, voiced_flag, voiced_probs, sr, hop_length=hop_length,
                     min_note_duration_frames=min_note_duration_frames,  
                     silence_threshold_frames=silence_threshold_frames,  
                     pitch_change_threshold_semitones=0.4,  
                     tuning_freq=args.tuning_freq,
                     pitch_average_window=pitch_average_window_frames)  

            # Plot if requested
            if args.plot or args.save_plots:
                save_path = args.save_plots
                plot_pitch_analysis(f0, voiced_flag, voiced_probs, sr, note_analyses, hop_length, save_path=save_path)

        except Exception as e:
            print(f"Error in pitch analysis: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    
    # --- VIBRATO ---
    if args.analysis_type in ["vibrato", "all"]:
        print("\nRunning vibrato analysis...")
        try:
            notes = segment_notes(
                f0, voiced_flag,
                min_note_duration_frames=min_note_duration_frames,
                silence_threshold_frames=silence_threshold_frames,
                pitch_change_threshold_semitones=0.5,
                pitch_average_window=pitch_average_window_frames
            )
            vibrato_results = analyze_vibrato(
                    f0, voiced_flag, voiced_probs, sr, 
                    tuning_freq=args.tuning_freq, hop_length=hop_length,
                    window_seconds=0.3,
                    step_seconds=0.1,
                    min_voiced_fraction=0.7
            )
            per_note_vib = aggregate_vibrato_per_note(notes, vibrato_results, sr, hop_length=hop_length)
            
            # print/plot continuous vibrato results
            print_vibrato_results(vibrato_results)
            if args.plot:
                print("\nGenerating vibrato plot...")
                plot_vibrato_analysis(vibrato_results)

            # print/plot segmented, per note vibrato results
            print_per_note_vibrato_results(per_note_vib)
            if args.plot:
                print("Generating per-note vibrato plot...")
                plot_per_note_vibrato(per_note_vib)
        except Exception as e:
            print(f"Error in vibrato analysis: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
    
    # --- TEMPERAMENT --- 
    if args.analysis_type in ["temperament", "all"]:
        print("\nRunning temperament analysis...")
        
        # Validate required arguments
        if not args.starting_note or not args.mode:
            print("Error: --starting-note and --mode are required for temperament analysis", file=sys.stderr)
            if args.analysis_type == "temperament":
                sys.exit(1)
            else:
                print("Skipping temperament analysis...")
        else:
            try:
                # Parse starting note
                from audio_analysis.temperament import parse_note_name
                tonic_pc, tonic_octave = parse_note_name(args.starting_note)
                
                temperament_results = analyze_temperament(
                    f0, voiced_flag, voiced_probs, sr,
                    tuning_freq=args.tuning_freq,
                    tonic_pc=tonic_pc,
                    tonic_octave=tonic_octave,
                    mode=args.mode
                )
                print_temperament_results(temperament_results)

                # Plot results if requested
                if args.plot or args.save_plots:
                    save_path = args.save_plots
                    plot_temperament_analysis(temperament_results, save_path=save_path)
            except Exception as e:
                print(f"Error in temperament analysis: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
