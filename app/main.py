import os

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.concurrency import run_in_threadpool

from app.analysis import run_note_analysis
from app.config import settings
from app.models import AnalyzeRequest, AnalyzeResponse, DatabaseEvent
from app.tasks import process_video_event

app = FastAPI(title="PracticeRoom Audio Analysis")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse, status_code=200)
async def analyze_audio(payload: AnalyzeRequest) -> AnalyzeResponse:
    if not os.path.isfile(payload.audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    try:
        result = await run_in_threadpool(run_note_analysis, payload.audio_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AnalyzeResponse(**result)


def _verify_webhook_secret(secret: str | None) -> None:
    expected = settings.webhook_secret
    if expected and secret != expected:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")


@app.post("/webhook/videos", status_code=202)
async def videos_webhook(
    payload: DatabaseEvent,
    background_tasks: BackgroundTasks,
    x_webhook_secret: str | None = Header(None),
) -> dict:
    """
    Webhook endpoint for Supabase database events on the videos table.
    Configure this URL in Supabase: Database -> Webhooks -> New Hook
    """
    _verify_webhook_secret(x_webhook_secret)
    background_tasks.add_task(process_video_event, payload)
    return {"status": "accepted"}