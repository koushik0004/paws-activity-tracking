import os
import torch
import numpy as np
from collections import deque
from typing import Tuple, Optional
from ultralytics import YOLO

try:
    from rtmlib import PoseTracker, RTMPose
    RTMLIB_AVAILABLE = True
except ImportError:
    RTMLIB_AVAILABLE = False

from .schemas import VisionPayload, AppearanceSchema, PostureSchema
from .vision import check_framing_margin, extract_dominant_color, classify_ap10k_posture


def resolve_device(preferred: Optional[str] = None) -> str:
    """Auto-selects Apple M-series MPS, NVIDIA CUDA, or CPU."""
    if preferred:
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class PawsVisionPipeline:
    """
    End-to-end computer vision and animal kinematic tracking pipeline.
    Combines YOLOv8 target detection, RTMPose (AP-10K), color analysis, and temporal pose hysteresis.
    """
    # Official OpenMMLab RTMPose-M trained on AP-10K (17 animal keypoints)
    DEFAULT_AP10K_ONNX = (
        "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
        "onnx_sdk/rtmpose-m_simcc-ap10k_pt-aic-coco_210e-256x256-7a041aa1_20230206.zip"
    )

    def __init__(
        self,
        detector_weights: str = "yolov8n.pt",
        pose_onnx_url_or_path: Optional[str] = None,
        use_animal_pose: bool = True,
        target_classes: list[int] = [16, 0],  # 16=dog, 0=person proxy
        conf_threshold: float = 0.50,
        framing_margin_pct: float = 0.10,
        velocity_deadband_px_s: float = 120.0,
        hold_gate_ms: int = 300,
        smooth_window: int = 12,
        min_confirm_frames: int = 5,
        fps: float = 25.0,
        kpt_conf_threshold: float = 0.30,
        device: Optional[str] = None,
    ):
        self.device = resolve_device(device)
        
        # Prefer compiled FP16 TensorRT detector on Jetson CUDA if present
        if self.device == "cuda" and os.path.exists("engines/yolov8n_fp16.engine"):
            detector_weights = "engines/yolov8n_fp16.engine"

        self.detector = YOLO(detector_weights, task="detect")
        self.target_classes = target_classes
        self.conf_threshold = conf_threshold
        self.kpt_conf_threshold = kpt_conf_threshold
        self.framing_margin_pct = framing_margin_pct
        self.velocity_deadband_px_s = velocity_deadband_px_s
        self.hold_gate_ms = hold_gate_ms
        self.min_confirm_frames = min_confirm_frames
        self.fps = fps if fps > 0 else 25.0

        # Initialize AP-10K Animal Pose Estimator
        self.pose_model = None
        self.pose_backend = "bbox_velocity_fallback"
        if use_animal_pose and RTMLIB_AVAILABLE:
            pose_src = pose_onnx_url_or_path or self.DEFAULT_AP10K_ONNX
            ort_backend = "tensorrt" if self.device == "cuda" else "onnxruntime"
            ort_device = "cuda" if self.device == "cuda" else "cpu"
            self.pose_model = RTMPose(
                pose_src,
                model_input_size=(256, 256),
                backend=ort_backend,
                device=ort_device,
            )
            self.pose_backend = f"ap10k_rtmpose_{ort_backend}"

        # State buffers
        self.center_history = deque(maxlen=smooth_window)
        self.pose_history = deque(maxlen=smooth_window)
        self.confirmed_pose: str = "UNKNOWN"
        self.candidate_pose: str = "UNKNOWN"
        self.candidate_count: int = 0
        self.last_stable_pose: Optional[str] = None
        self.stable_start_ms: Optional[int] = None
        self.frame_counter: int = 0
        self.cached_color: Optional[Tuple[Tuple[int, int, int], str]] = None

    def reset_state(self):
        """Resets temporal buffers between video clips or training sessions."""
        self.center_history.clear()
        self.pose_history.clear()
        self.confirmed_pose = "UNKNOWN"
        self.candidate_pose = "UNKNOWN"
        self.candidate_count = 0
        self.stable_start_ms = None
        self.last_stable_pose = None
        self.frame_counter = 0
        self.cached_color = None

    def process_frame(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> VisionPayload:
        """Runs end-to-end detection, AP-10K pose estimation, kinematics, and returns VisionPayload."""
        self.frame_counter += 1
        ts_ms = timestamp_ms if timestamp_ms is not None else int((self.frame_counter - 1) * (1000.0 / self.fps))
        h, w = frame.shape[:2]

        results = self.detector(frame, conf=self.conf_threshold, device=self.device, verbose=False)[0]

        # Select the LARGEST matching bounding box
        best_box = None
        best_cls_id = None
        best_area = 0
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id in self.target_classes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                area = max(0, x2 - x1) * max(0, y2 - y1)
                if area > best_area:
                    best_area = area
                    best_box = (x1, y1, x2, y2)
                    best_cls_id = cls_id

        if best_box is not None:
            x1, y1, x2, y2 = best_box
            framing_valid = check_framing_margin((x1, y1, x2, y2), w, h, self.framing_margin_pct)

            if self.cached_color is None:
                roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                self.cached_color = extract_dominant_color(roi)
            dom_rgb, dom_hex = self.cached_color

            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            self.center_history.append((cx, cy))
            if len(self.center_history) >= 2:
                prev_cx, prev_cy = self.center_history[0]
                dt_total = (len(self.center_history) - 1) / self.fps
                vel_px_s = float(np.hypot(cx - prev_cx, cy - prev_cy) / dt_total)
            else:
                vel_px_s = 0.0

            mean_kpt_conf = 0.0
            if self.pose_model is not None and best_cls_id == 16:
                keypoints, scores = self.pose_model(frame, bboxes=[[x1, y1, x2, y2]])
                raw_pose, mean_kpt_conf = classify_ap10k_posture(
                    keypoints[0], scores[0], (x1, y1, x2, y2), self.kpt_conf_threshold
                )
                if vel_px_s > self.velocity_deadband_px_s * 2.2:
                    raw_pose = "MOVING"
            else:
                raw_pose = "MOVING" if vel_px_s > self.velocity_deadband_px_s else "SITTING"

            # Temporal hysteresis
            self.pose_history.append(raw_pose)
            voted_pose = max(set(self.pose_history), key=self.pose_history.count)

            if self.confirmed_pose == "UNKNOWN":
                self.confirmed_pose = voted_pose
            elif voted_pose == self.confirmed_pose:
                self.candidate_pose = voted_pose
                self.candidate_count = 0
            else:
                if voted_pose == self.candidate_pose:
                    self.candidate_count += 1
                else:
                    self.candidate_pose = voted_pose
                    self.candidate_count = 1
                if self.candidate_count >= self.min_confirm_frames:
                    self.confirmed_pose = voted_pose
                    self.candidate_count = 0

            pose_state = self.confirmed_pose

            # Continuous hold gate (>= 300 ms) for stable rewardable poses
            if pose_state in ("SITTING", "STANDING", "DOWN", "PAW_LIFT"):
                if self.last_stable_pose == pose_state:
                    if self.stable_start_ms is None:
                        self.stable_start_ms = ts_ms
                    held_ms = ts_ms - self.stable_start_ms
                else:
                    self.last_stable_pose = pose_state
                    self.stable_start_ms = ts_ms
                    held_ms = 0
            else:
                self.last_stable_pose = None
                self.stable_start_ms = None
                held_ms = 0

            eligible = (held_ms >= self.hold_gate_ms) and framing_valid

            return VisionPayload(
                frame_id=self.frame_counter,
                timestamp_ms=ts_ms,
                subject_detected=True,
                framing_valid=framing_valid,
                appearance=AppearanceSchema(dominant_color_hex=dom_hex, dominant_rgb=dom_rgb),
                posture=PostureSchema(
                    state=pose_state,
                    held_ms=held_ms,
                    reward_eligible=eligible,
                    pose_backend=self.pose_backend,
                    mean_kpt_conf=mean_kpt_conf,
                ),
            )

        # Fallback when no subject is detected
        self.center_history.clear()
        self.pose_history.clear()
        self.confirmed_pose = "UNKNOWN"
        self.candidate_pose = "UNKNOWN"
        self.candidate_count = 0
        self.last_stable_pose = None
        self.stable_start_ms = None
        return VisionPayload(
            frame_id=self.frame_counter,
            timestamp_ms=ts_ms,
            subject_detected=False,
            framing_valid=False,
            appearance=AppearanceSchema(dominant_color_hex="#000000", dominant_rgb=(0, 0, 0)),
            posture=PostureSchema(
                state="UNKNOWN",
                held_ms=0,
                reward_eligible=False,
                pose_backend=self.pose_backend,
                mean_kpt_conf=0.0,
            ),
        )
