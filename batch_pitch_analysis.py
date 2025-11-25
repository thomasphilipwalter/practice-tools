#!/usr/bin/env python3
"""
Batch process audio files for pitch analysis and save plots.
"""

import os
import glob
import subprocess
import sys
from pathlib import Path

# Configuration
INPUT_DIR = "recordings/speed_tests"
OUTPUT_DIR = "results"
FILE_PATTERN = "*_4-npb.m4a"

# Analysis parameters
ANALYSIS_TYPE = "pitch"
TUNING_FREQ = 440
HOP_LENGTH = 512
SMOOTH = False

def main():
    # Create results directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all matching files
    pattern = os.path.join(INPUT_DIR, FILE_PATTERN)
    audio_files = sorted(glob.glob(pattern))
    
    if not audio_files:
        print(f"No files found matching {pattern}")
        return
    
    print(f"Found {len(audio_files)} files to process")
    print("-" * 80)
    
    for audio_file in audio_files:
        # Get base filename without extension
        base_name = Path(audio_file).stem
        
        print(f"\nProcessing: {audio_file}")
        
        # Build command
        cmd = [
            sys.executable, "-m", "audio_analysis.main",
            audio_file,
            "--analysis-type", ANALYSIS_TYPE,
            "--tuning-freq", str(TUNING_FREQ),
            "--hop-length", str(HOP_LENGTH),
            "--save-plots", os.path.join(OUTPUT_DIR, base_name)
        ]

        # Run analysis
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"  ✓ Completed: {base_name}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Error processing {base_name}: {e.stderr}")
    
    print("\n" + "=" * 80)
    print(f"Batch processing complete. Results saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()