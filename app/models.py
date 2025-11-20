from typing import List, Optional
from pydantic import BaseModel

class VideoRecord(BaseModel):
    id: str  # UUID as string
    title: str
    description: Optional[str] = None
    video_url: str
    user_id: str  # UUID as string
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class DatabaseEvent(BaseModel):
    type: str  # "INSERT", "UPDATE", etc.
    table: str  # "videos"
    schema_name: str
    record: VideoRecord
    old_record: Optional[VideoRecord] = None  # For UPDATE events

class AnalyzeRequest(BaseModel):
    audio_path: str

class NoteAnalysis(BaseModel):
    start_time: float
    end_time: float
    duration: float
    avg_f0: float
    nearest_note: str
    nearest_freq: float
    deviation_cents: float
    start_frame: int
    end_frame: int

class AnalyzeResponse(BaseModel):
    note_count: int
    notes: List[NoteAnalysis]