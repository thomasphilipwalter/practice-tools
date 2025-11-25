import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional

from supabase import create_client, Client

from app.config import settings

_supabase: Optional[Client] = None

def _get_supabase_client() -> Client:
    """Lazily initialize Supabase client only when needed"""
    global _supabase
    if _supabase is None:
        if not settings.supabase_url or not settings.supabase_service_key:
            raise ValueError("Supabase not configured. Set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables.")
        _supabase = create_client(settings.supabase_url, settings.supabase_service_key)
    return _supabase

def download_from_storage(bucket: str, object_path: str) -> str:
    client = _get_supabase_client()
    data = client.storage.from_(bucket).download(object_path)
    suffix = Path(object_path).suffix or ".dat"
    tmp = NamedTemporaryFile(delete=False, suffix=suffix, dir=settings.tmp_dir)
    tmp.write(data)
    tmp.flush()
    return tmp.name

def upsert_pitch_metrics(payload: Dict[str, Any]) -> None:
    client = _get_supabase_client()
    client.table(settings.pitch_metrics_table).upsert(payload).execute()

def upsert_vibrato_metrics(payload: Dict[str, Any]) -> None:
    client = _get_supabase_client()
    client.table(settings.vibrato_metrics_table).upsert(payload).execute()

def upsert_temperament_metrics(payload: Dict[str, Any]) -> None:
    client = _get_supabase_client()
    client.table(settings.temperament_metrics_table).upsert(payload).execute()