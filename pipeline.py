
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
        smooth_window: int = 5,
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
        self.last_stable_pose: Optional[str] = None
        self.stable_start_ms: Optional[int] = None
        self.frame_counter: int = 0
        self.cached_color: Optional[Tuple[Tuple[int, int, int], str]] = None

    def reset_state(self):
        """Resets temporal buffers between video clips or training sessions."""
        self.center_history.clear()
        self.pose_history.clear()
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

    def _classify_ap10k_posture(
        self, keypoints: np.ndarray, scores: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> Tuple[str, float]:
        """
        Classifies quadruped posture from 17 AP-10K keypoints normalized to bounding box:
        - STANDING: Torso horizontal (<28 deg) & hips elevated above floor plane
        - SITTING: Haunches/hips dropped near floor plane while shoulders/neck stay upright
        - DOWN: Both shoulders and hips dropped near floor plane
        - PAW_LIFT: Front paw lifted significantly while opposing front paw & hips stay grounded
        """

        x1, y1, x2, y2 = bbox
        box_w = max(1.0, float(x2 - x1))
        box_h = max(1.0, float(y2 - y1))

        # Helper to safely average visible left/right joint pairs normalized to [0..1] in bbox
        def avg_joint(idx_a: int, idx_b: int) -> Optional[np.ndarray]:
            pts = []
            for idx in (idx_a, idx_b):
                if scores[idx] >= self.kpt_conf_threshold:
                    nx = (keypoints[idx][0] - x1) / box_w
                    ny = (keypoints[idx][1] - y1) / box_h
                    pts.append([nx, ny])
            return np.mean(pts, axis=0) if pts else None

        mean_conf = float(np.mean(scores))
        shoulder = avg_joint(5, 8)      # left/right shoulder
        hip = avg_joint(11, 14)         # left/right hip (haunch)
        front_paw = avg_joint(7, 10)    # left/right front paw
        back_paw = avg_joint(13, 16)    # left/right back paw

        # Require core torso landmarks (shoulder + hip) to avoid guessing on occluded bodies
        if shoulder is None or hip is None:
            return "UNKNOWN", round(mean_conf, 3)

        # Estimate ground plane Y (bottom of visible paws or bottom of bbox = 1.0)
        paw_ys = [p[1] for p in (front_paw, back_paw) if p is not None]
        floor_y = max(paw_ys) if paw_ys else 0.95

        # Spine angle relative to horizontal (0 deg = flat/standing, >45 deg = upright sitting)
        dx = abs(shoulder[0] - hip[0]) + 1e-5
        dy = abs(hip[1] - shoulder[1])
        spine_angle_deg = float(np.degrees(np.arctan2(dy, dx)))

        # Normalized height above floor (smaller = closer to ground)
        hip_clearance = floor_y - hip[1]
        shoulder_clearance = floor_y - shoulder[1]

        # Check individual front paw lift (when both front paws are visible)
        if scores[7] >= self.kpt_conf_threshold and scores[10] >= self.kpt_conf_threshold:
            l_paw_y = (keypoints[7][1] - y1) / box_h
            r_paw_y = (keypoints[10][1] - y1) / box_h
            paw_delta_y = abs(l_paw_y - r_paw_y)
            if paw_delta_y > 0.18 and hip_clearance < 0.35:
                return "PAW_LIFT", round(mean_conf, 3)

        # Posture taxonomy rules (aligned with PAWS Buddy A2 Section 21)
        if shoulder_clearance < 0.28 and hip_clearance < 0.28:
            return "DOWN", round(mean_conf, 3)
        elif spine_angle_deg >= 38.0 and hip_clearance < 0.36:
            return "SITTING", round(mean_conf, 3)
        elif spine_angle_deg < 38.0 and hip_clearance >= 0.30:
            return "STANDING", round(mean_conf, 3)
        else:
            # Fallback to Sitting vs Standing by hip-to-shoulder vertical ratio
            return ("SITTING" if hip[1] > shoulder[1] + 0.18 else "STANDING"), round(mean_conf, 3)


    def process_frame(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> VisionPayload:
        """Runs end-to-end detection, AP-10K pose estimation, kinematics, and returns VisionPayload."""
        self.frame_counter += 1
        ts_ms = timestamp_ms if timestamp_ms is not None else int((self.frame_counter - 1) * (1000.0 / self.fps))
        h, w = frame.shape[:2]

        results = self.detector(frame, conf=self.conf_threshold, device=self.device, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id in self.target_classes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                framing_valid = self._check_framing_margin((x1, y1, x2, y2), w, h)

                # Extract & cache coat color on first valid detection
                if self.cached_color is None:
                    roi = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                    self.cached_color = self._extract_dominant_color(roi)
                dom_rgb, dom_hex = self.cached_color

                # Track global velocity to detect rapid walking/running
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                self.center_history.append((cx, cy))

                if len(self.center_history) >= 2:
                    prev_cx, prev_cy = self.center_history[0]
                    dt_total = (len(self.center_history) - 1) / self.fps
                    vel_px_s = float(np.hypot(cx - prev_cx, cy - prev_cy) / dt_total)
                else:
                    vel_px_s = 0.0

                mean_kpt_conf = 0.0
                if self.pose_model is not None and cls_id == 16:  # Quadruped pose on dog
                    keypoints, scores = self.pose_model(frame, bboxes=[[x1, y1, x2, y2]])
                    raw_pose, mean_kpt_conf = self._classify_ap10k_posture(
                        keypoints[0], scores[0], (x1, y1, x2, y2)
                    )
                    if vel_px_s > self.velocity_deadband_px_s * 1.5:
                        raw_pose = "MOVING"
                else:
                    # Fallback for human proxy (class 0) or when pose_model is disabled
                    raw_pose = "MOVING" if vel_px_s > self.velocity_deadband_px_s else "SITTING"

                # Majority-vote smoothing across rolling buffer to eliminate single-frame flicker
                self.pose_history.append(raw_pose)
                pose_state = max(set(self.pose_history), key=self.pose_history.count)
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
