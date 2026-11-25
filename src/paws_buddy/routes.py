import os
import time
import uuid
import cv2
import numpy as np
from typing import List, Optional
from fastapi import APIRouter, File, UploadFile, HTTPException, Query

from paws_buddy.pipeline import PawsVisionPipeline
from paws_buddy.schemas import (
    CueRequest,
    SessionState,
    CueEvaluationResponse,
    PostureSegment,
    StreamBenchmarkReport,
    VisionPayload,
)

router = APIRouter()

# Global training session state
current_session = SessionState()

# Shared vision engine reference (will be injected/initialized in main app)
vision_engine: Optional[PawsVisionPipeline] = None


def init_vision_engine(engine: PawsVisionPipeline):
    """Binds the active PawsVisionPipeline instance to the routes."""
    global vision_engine
    vision_engine = engine


def get_engine() -> PawsVisionPipeline:
    if vision_engine is None:
        raise HTTPException(status_code=500, detail="Vision engine is not initialized.")
    return vision_engine


def _now_ms() -> int:
    return int(time.time() * 1000)


@router.get("/health")
def health_check():
    engine = get_engine()
    eval_mode = os.getenv("EVAL_MODE", "true").lower() == "true"
    return {
        "status": "healthy",
        "device": engine.device,
        "eval_mode": eval_mode,
        "frame_counter": engine.frame_counter,
        "hold_gate_ms": engine.hold_gate_ms,
    }


@router.post("/reset")
def reset_session():
    engine = get_engine()
    engine.reset_state()
    return {"status": "reset_complete", "frame_counter": engine.frame_counter}


@router.post("/session/cue", response_model=SessionState)
def issue_training_cue(req: CueRequest):
    """Issues a fresh training cue with unique attempt_id and expiry window."""
    global current_session
    engine = get_engine()
    engine.reset_state()
    now = _now_ms()
    current_session = SessionState(
        attempt_id=f"att_{uuid.uuid4().hex[:8]}",
        status="ACTIVE",
        command_pose=req.command_pose,
        issued_at_ms=now,
        expires_at_ms=now + req.window_ms,
        reward_dispatched=False,
    )
    return current_session


@router.get("/session/status", response_model=SessionState)
def get_session_status():
    return current_session


@router.post("/session/evaluate_frame", response_model=CueEvaluationResponse)
async def evaluate_cue_frame(
    file: UploadFile = File(...),
    timestamp_ms: Optional[int] = Query(default=None),
):
    """Evaluates a frame against the active training cue (enforces 1 reward per attempt_id)."""
    global current_session
    engine = get_engine()
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")

    payload = engine.process_frame(frame, timestamp_ms=timestamp_ms)
    now = _now_ms()

    if current_session.status != "ACTIVE":
        return CueEvaluationResponse(
            session=current_session,
            vision=payload,
            trigger_esp32_toss=False,
            reason=f"Session is {current_session.status}; no active cue awaiting reward.",
        )

    if current_session.expires_at_ms and now > current_session.expires_at_ms:
        current_session.status = "EXPIRED"
        return CueEvaluationResponse(
            session=current_session,
            vision=payload,
            trigger_esp32_toss=False,
            reason="Command window expired before stable pose hold.",
        )

    pose_matches = (payload.posture.state == current_session.command_pose)
    if pose_matches and payload.posture.reward_eligible and not current_session.reward_dispatched:
        current_session.status = "MARKED"
        current_session.reward_dispatched = True
        return CueEvaluationResponse(
            session=current_session,
            vision=payload,
            trigger_esp32_toss=True,
            reason=f"Cue '{current_session.command_pose}' held >= {engine.hold_gate_ms}ms inside safe margin.",
        )

    return CueEvaluationResponse(
        session=current_session,
        vision=payload,
        trigger_esp32_toss=False,
        reason="Awaiting stable matching pose inside 10% margin.",
    )


@router.post("/predict", response_model=VisionPayload)
async def predict_frame(
    file: UploadFile = File(...),
    timestamp_ms: Optional[int] = Query(default=None),
):
    engine = get_engine()
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")

    return engine.process_frame(frame, timestamp_ms=timestamp_ms)


@router.post("/stream/evaluate", response_model=StreamBenchmarkReport)
def evaluate_stream(
    max_frames: int = Query(default=50, ge=1, le=500, description="Number of frames to benchmark"),
    video_path: Optional[str] = Query(default=None, description="Override video path when EVAL_MODE=true"),
    save_annotated: bool = Query(default=True, description="Save annotated debug MP4 to outputs/"),
):
    """Runs continuous frame ingestion, tracks per-frame latency, and exports debug video."""
    engine = get_engine()
    engine.reset_state()

    eval_mode = os.getenv("EVAL_MODE", "true").lower() == "true"
    default_video = os.getenv("TEST_VIDEO_PATH", "data/your_video_1.mp4")
    camera_index = int(os.getenv("LIVE_CAMERA_INDEX", "0"))

    source = (video_path or default_video) if eval_mode else camera_index

    if eval_mode and not os.path.exists(str(source)):
        raise HTTPException(status_code=404, detail=f"Test video not found: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail=f"Failed to open video source: {source}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    engine.fps = fps

    out_path = "outputs/local_stream_verified.mp4" if save_annotated else None
    writer = None
    if out_path:
        os.makedirs("outputs", exist_ok=True)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    latencies_ms: List[float] = []
    eligible_count = 0
    sample_f10: Optional[VisionPayload] = None
    posture_counts: dict[str, int] = {}
    segments: list[dict] = []

    for idx in range(1, max_frames + 1):
        ret, frame = cap.read()
        if not ret:
            break

        t_start = time.perf_counter()
        payload = engine.process_frame(frame)
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        latencies_ms.append(elapsed_ms)

        if payload.posture.reward_eligible:
            eligible_count += 1
        if idx == 10:
            sample_f10 = payload

        st = payload.posture.state
        ts = payload.timestamp_ms
        conf = payload.posture.mean_kpt_conf
        posture_counts[st] = posture_counts.get(st, 0) + 1

        if segments and segments[-1]["state"] == st:
            seg = segments[-1]
            seg["end_timestamp_ms"] = ts
            seg["held_ms"] = ts - seg["start_timestamp_ms"]
            seg["confs"].append(conf)
        else:
            segments.append({
                "segment_id": len(segments) + 1,
                "start_timestamp_ms": ts,
                "end_timestamp_ms": ts,
                "state": st,
                "held_ms": 0,
                "confs": [conf],
            })

        if writer:
            color = (0, 255, 0) if payload.posture.reward_eligible else (0, 200, 255)
            hud = f"F{idx:02d} | {payload.posture.state} ({payload.posture.held_ms}ms) | {elapsed_ms:.1f}ms"
            cv2.putText(frame, hud, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()

    if not latencies_ms:
        raise HTTPException(status_code=422, detail="No frames could be decoded from source.")

    steady_latencies = latencies_ms[1:] if len(latencies_ms) > 1 else latencies_ms
    avg_ms = float(np.mean(steady_latencies))
    max_ms = float(np.max(steady_latencies))

    posture_segments = [
        PostureSegment(
            segment_id=s["segment_id"],
            start_timestamp_ms=s["start_timestamp_ms"],
            end_timestamp_ms=s["end_timestamp_ms"],
            state=s["state"],
            held_ms=s["held_ms"],
            avg_kpt_conf=round(sum(s["confs"]) / len(s["confs"]), 3),
        )
        for s in segments
    ]

    return StreamBenchmarkReport(
        source=str(source),
        device=engine.device,
        frames_processed=len(latencies_ms),
        avg_latency_ms=round(avg_ms, 2),
        max_latency_ms=round(max_ms, 2),
        effective_fps=round(1000.0 / avg_ms, 1) if avg_ms > 0 else 0.0,
        meets_200ms_target=(max_ms <= 200.0),
        eligible_frames_count=eligible_count,
        posture_counts=posture_counts,
        postures=posture_segments,
        annotated_video_path=out_path,
        sample_frame_10=sample_f10,
    )
