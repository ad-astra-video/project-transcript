"""FastAPI server exposing an HTTP endpoint to launch the video
transcription/subtitle pipeline via POST request.

Run with:
uvicorn server.app:app --host 0.0.0.0 --port 8000

The endpoint accepts a JSON payload of the following form:
POST /audio-to-text
Content-Type: application/json
{
    "subscribe_url": "http://172.17.0.1:3389/sample",
    "control_url": "http://192.168.10.206:3389/sample",
    "publish_url": "http://172.17.0.1:3389/sample-output",
    "text_url": "http://172.17.0.1:3389/subtitles",
    "events_url": "http://172.17.0.1:3389/events",
    # optional
    # "params": {optional overrides
    #     "whisper_model": "medium",
    #     "hard_code_subtitles": true
    # }
    # }
    #
    # The server will spin up a background task running the pipeline with the
    # provided configuration and respond immediately with a 202 status and a
    # unique task_id.
    # """

from __future__ import annotations
import asyncio
import logging
import uuid
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from pipeline.config import PipelineConfig
from pipeline.main import VideoPipeline

logger = logging.getLogger("__name__")

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Audio-to-Text Pipeline API")
# ---------------------------------------------------------------------------
## Pydantic models
# ---------------------------------------------------------------------------
## For simplicity we accept params as a free-form dictionary and perform
# explicit validation against PipelineConfig inside the handler.
PipelineParamsDict = Dict[str, Any]


class AudioToTextRequest(BaseModel):
    subscribe_url: str = Field(..., description="Trickle subscribe URL")
    control_url: Optional[str] = Field(
        None, description="Control channel URL (reserved for future use)"
    )
    publish_url: str = Field(..., description="Trickle publish URL")
    text_url: Optional[str] = Field(
        None, description="Optional URL for posting generated subtitle files"
    )
    events_url: Optional[str] = Field(..., description="Optional URL for posting pipeline events")
    params: Optional[PipelineParamsDict] = Field(
        None, description="Overrides for PipelineConfig (mirrors PipelineConfig fields)"
    )

    @field_validator("subscribe_url", "publish_url", "text_url","events_url", check_fields=True)
    def _strip(cls, v):  # noqa: D401
        if isinstance(v, str):
            return v.strip()
        return v


class AudioToTextResponse(BaseModel):
    task_id: str
    status: str = "started"


# ---------------------------------------------------------------------------
# Task registry so we can keep track of running pipelines
# ---------------------------------------------------------------------------
_tasks: dict[str, asyncio.Task] = {}


async def _run_pipeline(cfg: PipelineConfig) -> None:
    """Initialize and run the pipeline until completion or error."""

    pipeline = VideoPipeline(cfg)
    try:
        await pipeline.initialize()
        await pipeline.run()
    except Exception as exc:  # pragma: no cover
        logger.error("Pipeline task failed: %s", exc)
    finally:
        logger.info("Pipeline task finished")
        # Gracefully shut down the FastAPI/uvicorn server once the pipeline ends
        try:
            import os, signal
            os.kill(os.getpid(), signal.SIGINT)  # uvicorn handles SIGINT gracefully
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to trigger server shutdown: %s", e)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/audio-to-text", response_model=AudioToTextResponse, status_code=202)
async def audio_to_text(request: AudioToTextRequest):
    # Build PipelineConfig instance
    params_dict: Dict[str, Any] = request.params or {}

    # Validate param names against PipelineConfig
    unknown_params = set(params_dict) - set(PipelineConfig.__annotations__)
    if unknown_params:
        raise HTTPException(status_code=400, detail=f"Unknown param fields: {unknown_params}")

    task_id = uuid.uuid4().hex
    cfg_kwargs = {
        **params_dict,
        "subscribe_url": request.subscribe_url,
        "publish_url": request.publish_url,
        "text_url": request.text_url,
        "events_url": request.events_url,
        "pipeline_uuid": task_id,
    }
    cfg = PipelineConfig(**cfg_kwargs)  # type: ignore[arg-type]

    task = asyncio.create_task(_run_pipeline(cfg))
    _tasks[task_id] = task

    # Automatically remove task from registry when done
    def _cleanup(t: asyncio.Task):
        _tasks.pop(task_id, None)

    task.add_done_callback(_cleanup)

    logger.info("Started pipeline task %s", task_id)

    return AudioToTextResponse(task_id=task_id)
