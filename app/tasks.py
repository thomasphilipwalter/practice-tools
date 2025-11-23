import logging
import os
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional
from urllib.parse import urlparse, unquote

from app.analysis import run_note_analysis
from app.config import settings
from app.models import DatabaseEvent
from app.storage import download_from_storage, upsert_metrics, _get_supabase_client

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
    return analysis_codes in [1, 4, 5, 7]

def _should_run_vibrato_analysis(analysis_codes: Optional[int]) -> bool:
    """
    Check if vibrato analysis should run based on analysis codes. 
    Vibrato analysis should run for codes: 2, 4, 6, 7
    """
    if analysis_codes is None:
        return False
    return analysis_codes in [2, 4, 6, 7]

def _should_run_temperament_analysis(analysis_codes: Optional[int]) -> bool:
    """
    Check if temperament analysis should run based on analysis codes. 
    Temperament analysis should run for codes: 3, 5, 6, 7
    """
    if analysis_codes is None:
        return False
    return analysis_codes in [3, 5, 6, 7]

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
        
        audio_path = None
        
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
        
        # Run PITCH analysis
        if _should_run_pitch_analysis(analysis_codes):
            logger.info(f"Running audio analysis...")
            result = run_note_analysis(audio_path, tuning_freq=tuning_freq)
            logger.info(f"Analysis complete: {result['note_count']} notes found")

            payload = {
                "video_id": video_id,
                "status": "completed",
                "note_count": result["note_count"],
                "notes": result["notes"],
                "avg_deviation": _calculate_avg_deviation(result["notes"]),
            }
            logger.info(f"Saving metrics to database...")
            upsert_metrics(payload)
            logger.info(f"Metrics saved successfully for video {video_id}")

        # Run VIBRATO analysis
        if _should_run_vibrato_analysis(analysis_codes):
            logger.info(f"Running vibrato analysis...")
            # TODO: implement functionality
        
        # Run TEMPERAMENT analysis
        if _should_run_temperament_analysis(analysis_codes):
            logger.info(f"Running temperament analysis...")
            # TODO: implement functionality
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}", exc_info=True)
        # Update status to failed if analysis fails
        try:
            client = _get_supabase_client()
            client.table(settings.metrics_table).upsert({
                "video_id": video_id,
                "status": "failed",
            }).execute()
            logger.info(f"Marked video {video_id} as failed in database")
        except Exception as db_error:
            logger.error(f"Failed to update status in database: {str(db_error)}")
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