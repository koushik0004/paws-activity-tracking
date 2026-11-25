from typing import Tuple, Optional, Literal
from pydantic import BaseModel

# ==========================================
# AP-10K 17-KEYPOINT QUADRUPED TOPOLOGY
# ==========================================
AP10K_KEYPOINTS = {
    0: "left_eye",
    1: "right_eye",
    2: "nose",
    3: "neck",
    4: "root_of_tail",
    5: "left_shoulder",
    6: "left_elbow",
    7: "left_front_paw",
    8: "right_shoulder",
    9: "right_elbow",
    10: "right_front_paw",
    11: "left_hip",
    12: "left_knee",
    13: "left_back_paw",
    14: "right_hip",
    15: "right_knee",
    16: "right_back_paw",
}

# ==========================================
# VISION INFERENCE SCHEMAS
# ==========================================
class AppearanceSchema(BaseModel):
    dominant_color_hex: str
    dominant_rgb: Tuple[int, int, int]


class PostureSchema(BaseModel):
    state: str
    held_ms: int
    reward_eligible: bool
    pose_backend: str = "bbox_velocity_fallback"
    mean_kpt_conf: float = 0.0


class VisionPayload(BaseModel):
    frame_id: int
    timestamp_ms: int
    subject_detected: bool
    framing_valid: bool
    appearance: AppearanceSchema
    posture: PostureSchema


# ==========================================
# TRAINING SESSION STATE SCHEMAS (CTRL-01)
# ==========================================
class CueRequest(BaseModel):
    command_pose: Literal["SITTING", "STANDING", "DOWN", "PAW_LIFT"] = "SITTING"
    window_ms: int = 5000


class SessionState(BaseModel):
    attempt_id: Optional[str] = None
    status: Literal["IDLE", "ACTIVE", "MARKED", "EXPIRED"] = "IDLE"
    command_pose: Optional[str] = None
    issued_at_ms: Optional[int] = None
    expires_at_ms: Optional[int] = None
    reward_dispatched: bool = False


class CueEvaluationResponse(BaseModel):
    session: SessionState
    vision: VisionPayload
    trigger_esp32_toss: bool
    reason: str


class PostureSegment(BaseModel):
    segment_id: int
    start_timestamp_ms: int
    end_timestamp_ms: int
    state: str
    held_ms: int
    avg_kpt_conf: float


class StreamBenchmarkReport(BaseModel):
    source: str
    device: str
    frames_processed: int
    avg_latency_ms: float
    max_latency_ms: float
    effective_fps: float
    meets_200ms_target: bool
    eligible_frames_count: int
    posture_counts: dict[str, int]
    postures: list[PostureSegment]
    annotated_video_path: Optional[str]
    sample_frame_10: Optional[VisionPayload]
