import logging
import os
import json

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import settings
from app.models import AnalyzeRequest, AnalyzeResponse, DatabaseEvent
from app.tasks import process_video_event

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

app = FastAPI(title="PracticeRoom Audio Analysis")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log validation errors with the actual request body."""
    body = await request.body()
    logger.error(f"Validation error: {exc.errors()}")
    logger.error(f"Request body: {body.decode() if body else 'No body'}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": body.decode() if body else None}
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def _verify_webhook_secret(secret: str | None) -> None:
    expected = settings.webhook_secret
    if expected and secret != expected:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")


@app.post("/webhook/videos", status_code=202)
async def videos_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_webhook_secret: str | None = Header(None),
) -> dict:
    """
        Webhook endpoint for Supabase video table events
    """
    logger.info(f"Webhook received - Method: {request.method}, Headers: {dict(request.headers)}")
    
    _verify_webhook_secret(x_webhook_secret)
    
    # Log raw request body for debugging
    body = await request.body()
    logger.info(f"Received webhook payload: {body.decode()}")
    
    try:
        payload_data = json.loads(body.decode())
        payload = DatabaseEvent(**payload_data)
    except Exception as e:
        logger.error(f"Failed to parse webhook payload: {e}")
        logger.error(f"Raw body: {body.decode()}")
        raise HTTPException(status_code=400, detail=f"Invalid payload: {str(e)}")
    
    background_tasks.add_task(process_video_event, payload)
    return {"status": "accepted"}