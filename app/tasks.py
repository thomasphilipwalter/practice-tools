import logging
import os
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional
from urllib.parse import urlparse, unquote

from app.analysis import run_pitch_analysis, run_vibrato_analysis, run_temperament_analysis
from audio_analysis.f0 import get_f0_contour
from app.config import settings
from app.models import DatabaseEvent
from app.storage import download_from_storage, _get_supabase_client, upsert_pitch_metrics, upsert_vibrato_metrics, upsert_temperament_metrics

logger = logging.getLogger(__name__)


def _extract_storage_path_from_url(video_url: str) -> tuple[str, str]:
    """
    Extract bucket and object path from Supabase storage public URL.
    Returns (bucket, object_path)
    """
    # SUPABASE URL STRUCT:
    # https://<project>.supabase.co/storage/v1/object/public/<bucket>/<path>
    parsed = urlparse(video_url)
    path_parts = parsed.path.split("/")
    
    # Find "public" in path and get bucket and object path
    try:
        public_idx = path_parts.index("public")
        bucket = path_parts[public_idx + 1]
        object_path = "/".join(path_parts[public_idx + 2:])
        object_path = unquote(object_path)  
        return bucket, object_path
    except (ValueError, IndexError):
        raise ValueError(f"Could not parse storage URL: {video_url}")


def extract_audio_from_video(video_path: str) -> str:
    """
    Extract the audio from a video file using ffmpeg.
    Returns path to temporary audio file.
    """
    audio_path = NamedTemporaryFile(delete=False, suffix=".wav", dir=settings.tmp_dir).name
    
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # PCM 16-bit audio
        "-ar", "44100",  # Sample rate
        "-ac", "2",  # Stereo
        "-y",  # Overwrite output file
        audio_path
    ]
    
    logger.info(f"Extracting audio from video: {video_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg failed: {result.stderr}")
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    
    logger.info(f"Audio extracted to: {audio_path}")
    return audio_path

def _should_run_pitch_analysis(analysis_codes: Optional[int]) -> bool:
    """
    Check if pitch analysis should run based on analysis codes. 
    Pitch analysis should run for codes: 1, 4, 5, 7
    """
    if analysis_codes is None:
        return False
    return analysis_codes in [1, 3, 5, 7]

def _should_run_vibrato_analysis(analysis_codes: Optional[int]) -> bool:
    """
    Check if vibrato analysis should run based on analysis codes. 
    Vibrato analysis should run for codes: 2, 4, 6, 7
    """
    if analysis_codes is None:
        return False
    return analysis_codes in [2, 3, 6, 7]

def _should_run_temperament_analysis(analysis_codes: Optional[int]) -> bool:
    """
    Check if temperament analysis should run based on analysis codes. 
    Temperament analysis should run for codes: 3, 5, 6, 7
    """
    if analysis_codes is None:
        return False
    return analysis_codes in [4, 5, 6, 7]

def process_video_event(event: DatabaseEvent) -> None:
    """
    Process a database event for a new video record.
    Downloads the video, extracts audio, analyzes, and saves metrics.
    """
    logger.info(f"Processing video event: {event.type} for video {event.record.id}")
    
    if event.type != "INSERT":
        logger.info(f"Skipping non-INSERT event: {event.type}")
        return
    
    video_record = event.record
    video_id = video_record.id
    analysis_codes = video_record.analysis_codes
    tuning_freq = video_record.tuning_freq or 440.0 # Default to 440 if not specified, although app should require specification
    
    if analysis_codes is None: # if none, then no analysis should run
        logger.info(f"Analysis not requested for {video_id} (analysis_codes={analysis_codes})")
        return

    try:
        # Extract storage bucket, path from video_url
        logger.info(f"Extracting storage path from URL: {video_record.video_url}")
        bucket, object_path = _extract_storage_path_from_url(video_record.video_url)
        logger.info(f"Bucket: {bucket}, Path: {object_path}")
        
        # Download video from storage
        logger.info(f"Downloading video from storage...")
        local_path = download_from_storage(bucket, object_path)
        logger.info(f"Video downloaded to: {local_path}")

        # Check if file is video
        video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
        file_ext = Path(object_path).suffix.lower()
        
        if file_ext in video_extensions:
            # Extract audio from video
            logger.info(f"Extracting audio from video file...")
            audio_path = extract_audio_from_video(local_path)
        else:
            # Assume already audio
            logger.info(f"File appears to be audio, using directly")
            audio_path = local_path

        # Extract f0 contour once (used by ALL analyssi types)
        logger.info(f"Extracting F0 contour from audio...")
        f0, voiced_flag, voiced_probs, sr, hop_length = get_f0_contour(audio_path)
        logger.info(f"F0 contour extracted: {len(f0)} frames")
        
        # Run PITCH analysis
        if _should_run_pitch_analysis(analysis_codes):
            logger.info(f"Running pitch analysis...")
            pitch_result = run_pitch_analysis(f0, voiced_flag, voiced_probs, sr, tuning_freq=tuning_freq, hop_length=hop_length)
            logger.info(f"Pitch analysis complete: {pitch_result['note_count']} notes found")

            payload = {
                "video_id": video_id,
                "status": "completed",
                "note_count": pitch_result["note_count"],
                "notes": pitch_result["notes"],
                "avg_deviation": _calculate_avg_deviation(pitch_result["notes"]),
            }
            logger.info(f"Saving pitch metrics to database...")
            upsert_pitch_metrics(payload)
            logger.info(f"Pitch metrics saved successfully for video {video_id}")

        # Run VIBRATO analysis
        if _should_run_vibrato_analysis(analysis_codes):
            logger.info(f"Running vibrato analysis...")
            vibrato_result = run_vibrato_analysis(f0, voiced_flag, voiced_probs, sr, tuning_freq=tuning_freq, hop_length=hop_length)
            logger.info(f"Vibrato analysis complete")

            payload = {
                "video_id": video_id,
                "status": "completed",
                "vibrato_values": vibrato_result,
            }
            logger.info(f"Saving vibrato metrics to database...")
            upsert_vibrato_metrics(payload)
            logger.info(f"Vibrato metrics saved successfully for video {video_id}")
        
        # Run TEMPERAMENT analysis
        if _should_run_temperament_analysis(analysis_codes):
            logger.info(f"Running temperament analysis...")
            # Validate required fields
            if not video_record.starting_note or not video_record.mode:
                logger.warning(f"Temperament analysis requires starting_note and mode. "
                             f"Starting note: {video_record.starting_note}, Mode: {video_record.mode}")
                logger.warning(f"Skipping temperament analysis for video {video_id}")
            else:
                temperament_result = run_temperament_analysis(
                    f0, voiced_flag, voiced_probs, sr,
                    tuning_freq=tuning_freq,
                    starting_note=video_record.starting_note,
                    mode=video_record.mode,
                    hop_length=hop_length
                )

                payload = {
                    "video_id": video_id,
                    "status": "completed",
                    "temperament_results": temperament_result,
                }
                logger.info(f"Saving temperament metrics to database...")
                upsert_temperament_metrics(payload)
                logger.info(f"Temperament metrics saved successfully for video {video_id}")
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}", exc_info=True)
        # Update status to failed if analysis fails
        try:
            if _should_run_pitch_analysis(analysis_codes):
                upsert_pitch_metrics({"video_id": video_id, "status": "failed"})
            if _should_run_vibrato_analysis(analysis_codes):
                upsert_vibrato_metrics({"video_id": video_id, "status": "failed"})
            if _should_run_temperament_analysis(analysis_codes):
                upsert_temperament_metrics({"video_id": video_id, "status": "failed"})
            logger.info(f"Marked video {video_id} analyses as failed in database")
        except Exception as db_error:
            logger.error(f"Failed to update status in database: {db_error}")
        raise
    finally:
        # Cleanup temp files
        logger.info("Cleaning up temporary files...")
        cleanup_paths = []
        if audio_path and audio_path != local_path:
            cleanup_paths.append(audio_path)
        if local_path:
            cleanup_paths.append(local_path)

        for path in cleanup_paths:
            try:
                os.remove(path)
                logger.info(f"Removed temp file: {path}")
            except OSError as e:
                logger.warning(f"Could not remove temp file {path}: {e}")       


def _calculate_avg_deviation(notes: list[Dict[str, Any]]) -> float | None:
    if not notes:
        return None
    values = [abs(note["deviation_cents"]) for note in notes]
    return float(sum(values) / len(values))