import numpy as np
import librosa
from scipy.signal import find_peaks

def aggregate_vibrato_per_note(notes, vib_result, sr, hop_length=512):
    """
    Aggregate sliding-window vibrato estimates into per-note summaries

    Parameters: 
    - notes: output from segment_notes (list of start and end frames
    - vib_result: output from analyze_vibrate (times, rate_hz, width_cents)
    - sr: sample rate
    - hop_length: hop_length used in pYIN for f0/frame times

    Returns
    per_note_vibrato : list of dict
        One dict per note:
        {
            "note_index": i,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_time,
            "end_time": end_time,
            "vibrato_rate_hz": mean_rate_or_nan,
            "vibrato_width_cents": mean_width_or_nan,
        }
    """

    # Create mapping between note frame boundaries to times
    times = librosa.frames_to_time(
        np.arange(max(n[1] for n in notes) + 1),
        sr=sr,
        hop_length=hop_length
    )

    # Extract vib results
    vib_times = vib_result["times"]
    vib_rate = vib_result["rate_hz"]
    vib_width = vib_result["width_cents"]

    results = []

    for idx, (start_frame, end_frame) in enumerate(notes):
        start_time = times[start_frame]
        end_time = times[end_frame]

        # windows whose center time falls inside this note
        in_note = (vib_times >= start_time) & (vib_times <= end_time)

        note_rates = vib_rate[in_note]
        note_widths = vib_width[in_note]

        # ignore NaNs
        note_rates = note_rates[~np.isnan(note_rates)]
        note_widths = note_widths[~np.isnan(note_widths)]

        if len(note_rates) == 0 or len(note_widths) == 0:
            mean_rate = np.nan
            mean_width = np.nan
        else:
            mean_rate = float(np.mean(note_rates))
            mean_width = float(np.mean(note_widths))

        results.append({
            "note_index": idx,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time": start_time,
            "end_time": end_time,
            "vibrato_rate_hz": mean_rate,
            "vibrato_width_cents": mean_width,
        })

    return results

def analyze_vibrato(f0, voiced_flag, voiced_probs, sr, hop_length=512, tuning_freq=440.0, window_seconds=0.6, step_seconds=0.1, min_voiced_fraction=0.6):
    """
    Analyze vibrato characteristics (rate and width) from F0 contour.
    """
    f0 = np.asarray(f0, dtype=float)
    voiced_flag = np.asarray(voiced_flag, dtype=bool)

    # frame rate of the f0 contour 
    frame_rate = sr / float(hop_length)
    T = len(f0)

    # time axis fro frames
    times = librosa.frames_to_time(np.arange(T), sr=sr, hop_length=hop_length)

    # Convert f0 to cents relative to tuning_freq
    cents = np.full_like(f0, np.nan, dtype=float)
    valid = (f0 > 0) & ~np.isnan(f0)
    cents[valid] = 1200 * np.log2(f0[valid] / tuning_freq)

    # window & step in frames
    win_frames = max(1, int(round(window_seconds * frame_rate))) # no. of f0 frames in each analysis window
    step_frames = max(1, int(round(step_seconds * frame_rate))) # no. of f0 frames to advance between windows

    # Storage for each analysis window
    out_times = []
    out_rate_hz = []
    out_width_cents = []

    for start in range(0, T - win_frames + 1, step_frames):
        end = start + win_frames
        window_cents = cents[start:end]
        window_voiced = voiced_flag[start:end]

        # Require enough voiced frames to trust vib estimate
        voiced_count = np.count_nonzero(window_voiced & ~np.isnan(window_cents))
        if voiced_count < min_voiced_fraction * win_frames:
            out_times.append(times[start + win_frames // 2]) # // 2 to assign center of window as time
            out_rate_hz.append(np.nan)
            out_width_cents.append(np.nan)
            continue
            
        # Require at least 3 valid samples for analysis (CHECK enough?)
        valid_mask = window_voiced & ~np.isnan(window_cents)
        if np.count_nonzero(valid_mask) < 3:
            out_times.append(times[start + win_frames // 2])
            out_rate_hz.append(np.nan)
            out_width_cents.append(np.nan)
            continue

        wc = window_cents[valid_mask]
        
        if len(wc) < 5:  # Need enough samples
            out_times.append(times[start + win_frames // 2])
            out_rate_hz.append(np.nan)
            out_width_cents.append(np.nan)
            continue

        # Better detrending - remove linear trend
        from scipy.signal import detrend, savgol_filter, welch
        
        wc_detrended = detrend(wc)
        
        # Smooth to reduce noise
        if len(wc_detrended) > 5:
            window_length = min(5, len(wc_detrended))
            if window_length >= 3 and window_length % 2 == 1:
                wc_detrended = savgol_filter(wc_detrended, window_length, 2)

        # Check for reasonable variation
        if np.nanstd(wc_detrended) < 1.0:
            out_times.append(times[start + win_frames // 2])
            out_rate_hz.append(0.0)
            out_width_cents.append(0.0)
            continue
            
        # --------- Vib rate: use FFT method (more robust) --------
        frequencies, psd = welch(wc_detrended, fs=frame_rate, nperseg=min(len(wc_detrended), 256))
        vib_range = (frequencies >= 2.0) & (frequencies <= 10.0)
        if np.any(vib_range) and np.max(psd[vib_range]) > 0:
            peak_idx = np.argmax(psd[vib_range])
            rate_hz = frequencies[vib_range][peak_idx]
            # Validate peak is significant
            if psd[vib_range][peak_idx] < np.mean(psd[vib_range]) * 1.5:
                rate_hz = np.nan
        else:
            rate_hz = np.nan

        # ------------ Vib width: with sanity check ----------
        peaks, _ = find_peaks(wc_detrended, height=np.std(wc_detrended) * 0.3)
        troughs, _ = find_peaks(-wc_detrended, height=np.std(wc_detrended) * 0.3)
        if len(peaks) > 0 and len(troughs) > 0:
            peak_vals = wc_detrended[peaks]
            trough_vals = wc_detrended[troughs]
            width_cents = (np.mean(peak_vals) - np.mean(trough_vals)) / 2.0
        else:
            width_cents = np.std(wc_detrended) * 2.0
        
        # Sanity check: vibrato width should be reasonable (< 50 cents typically)
        if width_cents > 100.0:
            width_cents = np.nan
            rate_hz = np.nan  # Invalidate both if width is absurd

        out_times.append(times[start + win_frames // 2])
        out_rate_hz.append(rate_hz)
        out_width_cents.append(width_cents)

    return {
        "times": np.array(out_times),
        "rate_hz": np.array(out_rate_hz),
        "width_cents": np.array(out_width_cents),
    }

    return {}
    