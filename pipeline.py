
import os

import cv2
import torch
import numpy as np
from collections import deque
from typing import Tuple, Optional
from pydantic import BaseModel
from ultralytics import YOLO
from sklearn.cluster import KMeans

try:
    from rtmlib import PoseTracker, RTMPose
    RTMLIB_AVAILABLE = True
except ImportError:
    RTMLIB_AVAILABLE = False

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
# STRICT PYDANTIC OUTPUT SCHEMA
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

def resolve_device(preferred: Optional[str] = None) -> str:
    """Auto-selects Apple M4 MPS, NVIDIA CUDA, or CPU."""
    if preferred:
        return preferred
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ==========================================
# MODULAR COMPUTER VISION PIPELINE CLASS
# ==========================================
class PawsVisionPipeline:
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
        smooth_window: int = 12,               # NEW: Increased from 5 -> 12 frames (~400ms) to stop flicker
        min_confirm_frames: int = 5,           # NEW: Require 5 consecutive frames (~165ms) to switch pose
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
        self.min_confirm_frames = min_confirm_frames  # NEW: Store hysteresis threshold
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
        self.confirmed_pose: str = "UNKNOWN"   # NEW: Smoothed pose after hysteresis
        self.candidate_pose: str = "UNKNOWN"   # NEW: Incoming pose being verified
        self.candidate_count: int = 0          # NEW: Consecutive frame counter for candidate_pose
        self.last_stable_pose: Optional[str] = None
        self.stable_start_ms: Optional[int] = None
        self.frame_counter: int = 0
        self.cached_color: Optional[Tuple[Tuple[int, int, int], str]] = None

    def reset_state(self):
        """Resets temporal buffers between video clips or training sessions."""
        self.center_history.clear()
        self.pose_history.clear()
        self.confirmed_pose = "UNKNOWN"        # NEW: Reset hysteresis state
        self.candidate_pose = "UNKNOWN"        # NEW: Reset candidate state
        self.candidate_count = 0               # NEW: Reset counter
        self.stable_start_ms = None
        self.last_stable_pose = None
        self.frame_counter = 0
        self.cached_color = None

    def _check_framing_margin(self, bbox: Tuple[int, int, int, int], w: int, h: int) -> bool:
        x1, y1, x2, y2 = bbox
        min_x, min_y = w * self.framing_margin_pct, h * self.framing_margin_pct
        max_x, max_y = w * (1.0 - self.framing_margin_pct), h * (1.0 - self.framing_margin_pct)
        return (x1 >= min_x) and (y1 >= min_y) and (x2 <= max_x) and (y2 <= max_y)

    def _extract_dominant_color(self, roi_bgr: np.ndarray, k: int = 2) -> Tuple[Tuple[int, int, int], str]:
        h, w = roi_bgr.shape[:2]
        if h < 10 or w < 10:
            return (0, 0, 0), "#000000"
        center_crop = roi_bgr[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]
        small_rgb = cv2.cvtColor(cv2.resize(center_crop, (64, 64)), cv2.COLOR_BGR2RGB)
        pixels = small_rgb.reshape(-1, 3)

        kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
        labels = kmeans.fit_predict(pixels)
        dominant_idx = np.bincount(labels).argmax()
        rgb = kmeans.cluster_centers_[dominant_idx].astype(int)
        rgb_tuple = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        hex_code = "#{:02X}{:02X}{:02X}".format(*rgb_tuple)
        return rgb_tuple, hex_code

    @staticmethod
    def _joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """Computes interior angle (degrees) at joint b formed by points a-b-c."""
        ba = a - b
        bc = c - b
        denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
        cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))
    
    def _classify_ap10k_posture(
        self, keypoints: np.ndarray, scores: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[str, float]:
        """
        Perspective-invariant posture classifier using limb elevation above the floor plane:
        - STANDING: Elbows & knees lifted well above paws (vertical forearm & shin segments)
        - CROUCHING: Knees/elbows partially lowered (transitional crouch/hunch)
        - DOWN: Elbows & knees resting flat near paw floor level
        - SITTING: Front forearms upright while hips/knees drop to floor level
        - PAW_LIFT: One forepaw elevated significantly higher while stationary
        """
        x1, y1, x2, y2 = bbox
        box_w = max(1.0, float(x2 - x1))
        box_h = max(1.0, float(y2 - y1))

        def avg_joint(idx_a: int, idx_b: int) -> Optional[np.ndarray]:
            pts = []
            for idx in (idx_a, idx_b):
                if scores[idx] >= self.kpt_conf_threshold:
                    pts.append([(keypoints[idx][0] - x1) / box_w, (keypoints[idx][1] - y1) / box_h])
            return np.mean(pts, axis=0) if pts else None

        mean_conf = float(np.mean(scores))
        shoulder = avg_joint(5, 8)
        elbow = avg_joint(6, 9)
        front_paw = avg_joint(7, 10)
        hip = avg_joint(11, 14)
        knee = avg_joint(12, 15)
        back_paw = avg_joint(13, 16)

        if shoulder is None or hip is None:
            return "UNKNOWN", round(mean_conf, 3)

        # NEW 1: Measure vertical height of elbows & knees above their paws (0.0 = flat on floor)
        forearm_lift = (front_paw[1] - elbow[1]) if (front_paw is not None and elbow is not None) else 0.12
        shin_lift = (back_paw[1] - knee[1]) if (back_paw is not None and knee is not None) else 0.12
        hip_minus_shoulder_y = hip[1] - shoulder[1]

        fore_angle = (
            self._joint_angle(shoulder, elbow, front_paw)
            if (elbow is not None and front_paw is not None)
            else 145.0
        )
        hind_angle = (
            self._joint_angle(hip, knee, back_paw)
            if (knee is not None and back_paw is not None)
            else 145.0
        )

        # NEW 2: PAW_LIFT — one front paw lifted noticeably higher while front legs are upright
        if forearm_lift > 0.14 and scores[7] >= self.kpt_conf_threshold and scores[10] >= self.kpt_conf_threshold:
            l_paw_y = (keypoints[7][1] - y1) / box_h
            r_paw_y = (keypoints[10][1] - y1) / box_h
            if abs(l_paw_y - r_paw_y) > 0.20:
                return "PAW_LIFT", round(mean_conf, 3)

        # NEW 3: DOWN — elbows and knees collapsed onto the floor plane (even if facing camera)
        if forearm_lift < 0.09 and shin_lift < 0.10:
            return "DOWN", round(mean_conf, 3)

        # NEW 4: SITTING — upright front forearms, low hips, folded hind legs
        if forearm_lift >= 0.13 and hip_minus_shoulder_y > 0.15 and hind_angle < 115.0:
            return "SITTING", round(mean_conf, 3)

        # NEW 5: CROUCHING — intermediate limb bend / lowered hindquarters between STANDING and DOWN
        if (0.08 <= shin_lift <= 0.15 or 0.08 <= forearm_lift <= 0.14) and (
            hind_angle < 135.0 or fore_angle < 135.0
        ):
            return "CROUCHING", round(mean_conf, 3)

        # NEW 6: STANDING — elbows and knees clearly elevated off the ground
        if forearm_lift >= 0.13 and shin_lift >= 0.12:
            return "STANDING", round(mean_conf, 3)

        return "DOWN" if (forearm_lift + shin_lift) < 0.20 else "CROUCHING", round(mean_conf, 3)


    def process_frame(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> VisionPayload:
        """Runs end-to-end detection, AP-10K pose estimation, kinematics, and returns VisionPayload."""
        self.frame_counter += 1
        ts_ms = timestamp_ms if timestamp_ms is not None else int((self.frame_counter - 1) * (1000.0 / self.fps))
        h, w = frame.shape[:2]

        results = self.detector(frame, conf=self.conf_threshold, device=self.device, verbose=False)[0]

        # NEW 1: Select the LARGEST matching bounding box (prevents swapping between foreground dog & background dog)
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
            framing_valid = self._check_framing_margin((x1, y1, x2, y2), w, h)

            if self.cached_color is None:
                roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                self.cached_color = self._extract_dominant_color(roi)
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
                raw_pose, mean_kpt_conf = self._classify_ap10k_posture(
                    keypoints[0], scores[0], (x1, y1, x2, y2)
                )
                # NEW 2: Raise fast-walk threshold (2.2x deadband) so slow walking stays classified as STANDING
                if vel_px_s > self.velocity_deadband_px_s * 2.2:
                    raw_pose = "MOVING"
            else:
                raw_pose = "MOVING" if vel_px_s > self.velocity_deadband_px_s else "SITTING"

            # NEW 3: Temporal hysteresis — require min_confirm_frames consecutive frames before switching pose
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
