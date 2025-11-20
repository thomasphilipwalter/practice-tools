import os
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict
from urllib.parse import urlparse, unquote

from app.analysis import run_note_analysis
from app.config import settings
from app.models import DatabaseEvent
from app.storage import download_from_storage, upsert_metrics, _get_supabase_client


def _extract_storage_path_from_url(video_url: str) -> tuple[str, str]:
    """
    Extract bucket and object path from Supabase storage public URL.
    Returns (bucket, object_path)
    """
    # Supabase storage URLs look like:
    # https://<project>.supabase.co/storage/v1/object/public/<bucket>/<path>
    parsed = urlparse(video_url)
    path_parts = parsed.path.split("/")
    
    # Find "public" in the path and get bucket and object path
    try:
        public_idx = path_parts.index("public")
        bucket = path_parts[public_idx + 1]
        object_path = "/".join(path_parts[public_idx + 2:])
        object_path = unquote(object_path)  # URL decode
        return bucket, object_path
    except (ValueError, IndexError):
        raise ValueError(f"Could not parse storage URL: {video_url}")


def extract_audio_from_video(video_path: str) -> str:
    """
    Extract audio from video file using ffmpeg.
    Returns path to temporary audio file.
    """
    audio_path = NamedTemporaryFile(delete=False, suffix=".wav", dir=settings.tmp_dir).name
    
    # Use ffmpeg to extract audio
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
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    
    return audio_path


def process_video_event(event: DatabaseEvent) -> None:
    """
    Process a database event for a new video record.
    Downloads the video, extracts audio, analyzes, and saves metrics.
    """
    if event.type != "INSERT":
        # Only process new video insertions
        return
    
    video_record = event.record
    video_id = video_record.id
    
    # Extract storage bucket and path from video_url
    bucket, object_path = _extract_storage_path_from_url(video_record.video_url)
    
    # Download video from storage
    local_path = download_from_storage(bucket, object_path)
    audio_path = None
    
    try:
        # Check if file is video (by extension)
        video_extensions = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}
        file_ext = Path(object_path).suffix.lower()
        
        if file_ext in video_extensions:
            # Extract audio from video
            audio_path = extract_audio_from_video(local_path)
        else:
            # Assume it's already audio
            audio_path = local_path
        
        # Run analysis
        result = run_note_analysis(audio_path)
        
        # Save metrics to database
        payload = {
            "video_id": video_id,
            "status": "completed",
            "note_count": result["note_count"],
            "notes": result["notes"],
            "avg_deviation": _calculate_avg_deviation(result["notes"]),
        }
        upsert_metrics(payload)
    except Exception as e:
        # Update status to failed if analysis fails
        try:
            client = _get_supabase_client()
            client.table(settings.metrics_table).upsert({
                "video_id": video_id,
                "status": "failed",
            }).execute()
        except Exception:
            pass
        raise
    finally:
        # Cleanup temp files
        for path in [local_path, audio_path]:
            if path and path != local_path:
                try:
                    os.remove(path)
                except OSError:
                    pass
        try:
            os.remove(local_path)
        except OSError:
            pass


def _calculate_avg_deviation(notes: list[Dict[str, Any]]) -> float | None:
    if not notes:
        return None
    values = [abs(note["deviation_cents"]) for note in notes]
    return float(sum(values) / len(values))