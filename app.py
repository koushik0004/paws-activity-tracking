"""
Root entrypoint shim preserving backward compatibility for `uvicorn app:app` and `Makefile`.
Directly re-exports from `src.paws_buddy`.
"""

from paws_buddy.app import app, create_app
from paws_buddy.pipeline import PawsVisionPipeline
from paws_buddy.schemas import (
    CueRequest,
    SessionState,
    CueEvaluationResponse,
    PostureSegment,
    StreamBenchmarkReport,
    VisionPayload,
)
from paws_buddy.routes import (
    current_session,
    vision_engine,
    health_check,
    reset_session,
    issue_training_cue,
    get_session_status,
    evaluate_cue_frame,
    predict_frame,
    evaluate_stream,
)

__all__ = [
    "app",
    "create_app",
    "PawsVisionPipeline",
    "VisionPayload",
    "CueRequest",
    "SessionState",
    "CueEvaluationResponse",
    "PostureSegment",
    "StreamBenchmarkReport",
    "current_session",
    "vision_engine",
    "health_check",
    "reset_session",
    "issue_training_cue",
    "get_session_status",
    "evaluate_cue_frame",
    "predict_frame",
    "evaluate_stream",
]
