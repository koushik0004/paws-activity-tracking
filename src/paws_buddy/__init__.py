"""PAWS Buddy - Vision inference and pet training microservice."""

from .schemas import (
    VisionPayload,
    AppearanceSchema,
    PostureSchema,
    CueRequest,
    SessionState,
    CueEvaluationResponse,
    PostureSegment,
    StreamBenchmarkReport,
    AP10K_KEYPOINTS,
)
from .pipeline import PawsVisionPipeline, resolve_device
from .vision import check_framing_margin, extract_dominant_color, classify_ap10k_posture
from .app import app, create_app, main

__all__ = [
    "VisionPayload",
    "AppearanceSchema",
    "PostureSchema",
    "CueRequest",
    "SessionState",
    "CueEvaluationResponse",
    "PostureSegment",
    "StreamBenchmarkReport",
    "AP10K_KEYPOINTS",
    "PawsVisionPipeline",
    "resolve_device",
    "check_framing_margin",
    "extract_dominant_color",
    "classify_ap10k_posture",
    "app",
    "create_app",
    "main",
]
