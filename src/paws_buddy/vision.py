import os
import cv2
import numpy as np
from typing import Tuple, Optional
from sklearn.cluster import KMeans

# AP-10K 17-Keypoint Quadruped Topology Index Mapping
# 5: left_shoulder, 6: left_elbow, 7: left_front_paw
# 8: right_shoulder, 9: right_elbow, 10: right_front_paw
# 11: left_hip, 12: left_knee, 13: left_back_paw
# 14: right_hip, 15: right_knee, 16: right_back_paw


def check_framing_margin(
    bbox: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    margin_pct: float = 0.10,
) -> bool:
    """Verifies that the bounding box sits strictly within the safe framing margin."""
    x1, y1, x2, y2 = bbox
    min_x, min_y = frame_width * margin_pct, frame_height * margin_pct
    max_x, max_y = frame_width * (1.0 - margin_pct), frame_height * (1.0 - margin_pct)
    return (x1 >= min_x) and (y1 >= min_y) and (x2 <= max_x) and (y2 <= max_y)


def extract_dominant_color(roi_bgr: np.ndarray, k: int = 2) -> Tuple[Tuple[int, int, int], str]:
    """Extracts dominant RGB and hex code using K-Means clustering on the central ROI crop."""
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


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Computes interior angle (degrees) at joint b formed by points a-b-c."""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cos_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def classify_ap10k_posture(
    keypoints: np.ndarray,
    scores: np.ndarray,
    bbox: Tuple[int, int, int, int],
    kpt_conf_threshold: float = 0.30,
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
            if scores[idx] >= kpt_conf_threshold:
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

    forearm_lift = (front_paw[1] - elbow[1]) if (front_paw is not None and elbow is not None) else 0.12
    shin_lift = (back_paw[1] - knee[1]) if (back_paw is not None and knee is not None) else 0.12
    hip_minus_shoulder_y = hip[1] - shoulder[1]

    fore_angle = (
        joint_angle(shoulder, elbow, front_paw)
        if (elbow is not None and front_paw is not None)
        else 145.0
    )
    hind_angle = (
        joint_angle(hip, knee, back_paw)
        if (knee is not None and back_paw is not None)
        else 145.0
    )

    # 1. PAW_LIFT — one front paw lifted noticeably higher while front legs are upright
    if forearm_lift > 0.14 and scores[7] >= kpt_conf_threshold and scores[10] >= kpt_conf_threshold:
        l_paw_y = (keypoints[7][1] - y1) / box_h
        r_paw_y = (keypoints[10][1] - y1) / box_h
        if abs(l_paw_y - r_paw_y) > 0.20:
            return "PAW_LIFT", round(mean_conf, 3)

    # 2. DOWN — elbows and knees collapsed onto the floor plane
    if forearm_lift < 0.09 and shin_lift < 0.10:
        return "DOWN", round(mean_conf, 3)

    # 3. SITTING — upright front forearms, low hips, folded hind legs
    if forearm_lift >= 0.13 and hip_minus_shoulder_y > 0.15 and hind_angle < 115.0:
        return "SITTING", round(mean_conf, 3)

    # 4. CROUCHING — intermediate limb bend between STANDING and DOWN
    if (0.08 <= shin_lift <= 0.15 or 0.08 <= forearm_lift <= 0.14) and (
        hind_angle < 135.0 or fore_angle < 135.0
    ):
        return "CROUCHING", round(mean_conf, 3)

    # 5. STANDING — elbows and knees clearly elevated off the ground
    if forearm_lift >= 0.13 and shin_lift >= 0.12:
        return "STANDING", round(mean_conf, 3)

    return "DOWN" if (forearm_lift + shin_lift) < 0.20 else "CROUCHING", round(mean_conf, 3)
