"""
Root entrypoint shim preserving backward compatibility for `pipeline.py`.
Directly re-exports from `src.paws_buddy`.
"""

from paws_buddy.schemas import (
    AP10K_KEYPOINTS,
    AppearanceSchema,
    PostureSchema,
    VisionPayload,
)
from paws_buddy.vision import (
    check_framing_margin,
    extract_dominant_color,
    joint_angle,
    classify_ap10k_posture,
)
from paws_buddy.pipeline import PawsVisionPipeline, resolve_device

__all__ = [
    "AP10K_KEYPOINTS",
    "AppearanceSchema",
    "PostureSchema",
    "VisionPayload",
    "PawsVisionPipeline",
    "resolve_device",
    "check_framing_margin",
    "extract_dominant_color",
    "joint_angle",
    "classify_ap10k_posture",
]
