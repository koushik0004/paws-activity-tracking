
import cv2
import torch
import numpy as np
from collections import deque
from typing import Tuple, Optional
from pydantic import BaseModel
from ultralytics import YOLO
from sklearn.cluster import KMeans


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
    def __init__(
        self,
        detector_weights: str = "yolov8n.pt",
        target_classes: list[int] = [16, 0],  # 16=dog, 0=person proxy
        conf_threshold: float = 0.50,
        framing_margin_pct: float = 0.10,
        velocity_deadband_px_s: float = 120.0,
        hold_gate_ms: int = 300,
        smooth_window: int = 5,
        fps: float = 25.0,
        device: Optional[str] = None,
    ):
        self.device = resolve_device(device)
        self.detector = YOLO(detector_weights)
        self.target_classes = target_classes
        self.conf_threshold = conf_threshold
        self.framing_margin_pct = framing_margin_pct
        self.velocity_deadband_px_s = velocity_deadband_px_s
        self.hold_gate_ms = hold_gate_ms
        self.fps = fps if fps > 0 else 25.0

        # State buffers
        self.center_history = deque(maxlen=smooth_window)
        self.stable_start_ms: Optional[int] = None
        self.frame_counter: int = 0
        self.cached_color: Optional[Tuple[Tuple[int, int, int], str]] = None

    def reset_state(self):
        """Resets temporal buffers between video clips or training sessions."""
        self.center_history.clear()
        self.stable_start_ms = None
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

    def process_frame(self, frame: np.ndarray, timestamp_ms: Optional[int] = None) -> VisionPayload:
        """Runs end-to-end detection, framing check, kinematics, and returns VisionPayload."""
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

                # Smoothed center velocity
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                self.center_history.append((cx, cy))

                if len(self.center_history) >= 2:
                    prev_cx, prev_cy = self.center_history[0]
                    dt_total = (len(self.center_history) - 1) / self.fps
                    vel_px_s = float(np.hypot(cx - prev_cx, cy - prev_cy) / dt_total)
                else:
                    vel_px_s = 0.0

                pose_state = "MOVING" if vel_px_s > self.velocity_deadband_px_s else "SITTING"

                if pose_state == "SITTING":
                    if self.stable_start_ms is None:
                        self.stable_start_ms = ts_ms
                    held_ms = ts_ms - self.stable_start_ms
                else:
                    self.stable_start_ms = None
                    held_ms = 0

                eligible = (held_ms >= self.hold_gate_ms) and framing_valid

                return VisionPayload(
                    frame_id=self.frame_counter,
                    timestamp_ms=ts_ms,
                    subject_detected=True,
                    framing_valid=framing_valid,
                    appearance=AppearanceSchema(dominant_color_hex=dom_hex, dominant_rgb=dom_rgb),
                    posture=PostureSchema(state=pose_state, held_ms=held_ms, reward_eligible=eligible),
                )

        # Fallback when no subject is detected
        self.center_history.clear()
        self.stable_start_ms = None
        return VisionPayload(
            frame_id=self.frame_counter,
            timestamp_ms=ts_ms,
            subject_detected=False,
            framing_valid=False,
            appearance=AppearanceSchema(dominant_color_hex="#000000", dominant_rgb=(0, 0, 0)),
            posture=PostureSchema(state="UNKNOWN", held_ms=0, reward_eligible=False),
        )
