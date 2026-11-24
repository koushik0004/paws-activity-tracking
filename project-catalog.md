# PAWS Buddy Project Catalog

## Index of Classes & Data Models

### Vision Pipeline Schemas ([pipeline.py](file:///Users/koushiksadhukhan/projects/paws-buddy/pipeline.py))
- **`AppearanceSchema`** (`pydantic.BaseModel`):
  - `dominant_color_hex`: Hex color string representing the coat (e.g. `"#8A6543"`).
  - `dominant_rgb`: Tuple of `(R, G, B)` integer values.
- **`PostureSchema`** (`pydantic.BaseModel`):
  - `state`: Detected pose state (`"SITTING"`, `"MOVING"`, `"UNKNOWN"`).
  - `held_ms`: Milliseconds the posture has been continuously held without moving.
  - `reward_eligible`: Boolean flag indicating if posture is held $\ge 300\text{ms}$ and inside the safe framing margin.
- **`VisionPayload`** (`pydantic.BaseModel`):
  - `frame_id`: Incremental integer sequence counter.
  - `timestamp_ms`: Calculated or provided frame timestamp in ms.
  - `subject_detected`: Flag whether target class (dog / person proxy) was identified.
  - `framing_valid`: True if bounding box is strictly within framing margin percentage (default 10%).
  - `appearance`: Instance of `AppearanceSchema`.
  - `posture`: Instance of `PostureSchema`.

### State Machine Schemas ([app.py](file:///Users/koushiksadhukhan/projects/paws-buddy/app.py))
- **`CueRequest`** (`pydantic.BaseModel`):
  - `command_pose`: Target posture command (`"SITTING"`, `"STANDING"`, `"DOWN"`, `"PAW_LIFT"`). Default: `"SITTING"`.
  - `window_ms`: Duration in milliseconds before cue expires (default: 5000ms).
- **`SessionState`** (`pydantic.BaseModel`):
  - `attempt_id`: Unique attempt ID (`att_<uuid>`).
  - `status`: Session lifecycle state (`"IDLE"`, `"ACTIVE"`, `"MARKED"`, `"EXPIRED"`).
  - `command_pose`: Active command pose.
  - `issued_at_ms`: Epoch timestamp in ms when cue was issued.
  - `expires_at_ms`: Epoch timestamp in ms when cue will expire.
  - `reward_dispatched`: Guard flag ensuring at most one reward per attempt.
- **`CueEvaluationResponse`** (`pydantic.BaseModel`):
  - `session`: Current `SessionState`.
  - `vision`: Extracted `VisionPayload`.
  - `trigger_esp32_toss`: Boolean indicating whether hardware treat tosser should trigger.
  - `reason`: Explanatory human-readable string explaining the decision.
- **`StreamBenchmarkReport`** (`pydantic.BaseModel`):
  - Benchmarking statistics (`avg_latency_ms`, `max_latency_ms`, `effective_fps`, `meets_200ms_target`, `annotated_video_path`, etc.).

---

## Core Classes & Methods

### `PawsVisionPipeline` ([pipeline.py](file:///Users/koushiksadhukhan/projects/paws-buddy/pipeline.py))
- `__init__(detector_weights, target_classes, conf_threshold, framing_margin_pct, velocity_deadband_px_s, hold_gate_ms, smooth_window, fps, device)`: Initializes YOLO model and temporal smoothing buffers.
- `reset_state()`: Clears center history, stable start timestamps, and cached coat colors.
- `_check_framing_margin(bbox, w, h)`: Computes boundary margins and checks if bounding box stays inside `[margin, 1 - margin]`.
- `_extract_dominant_color(roi_bgr, k)`: Runs K-Means clustering on central crop of bounding box to identify predominant coat color.
- `process_frame(frame, timestamp_ms)`: Executes object detection, kinematics smoothing, pose determination, and returns `VisionPayload`.

---

## API Endpoints ([app.py](file:///Users/koushiksadhukhan/projects/paws-buddy/app.py))

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System status, current device (`mps`/`cpu`), frame counter, and parameters |
| `POST` | `/reset` | Resets vision pipeline temporal buffers |
| `POST` | `/session/cue` | Issues a new training cue (`CueRequest`) and transitions session to `ACTIVE` |
| `GET` | `/session/status` | Queries the current training session state |
| `POST` | `/session/evaluate_frame` | Evaluates an uploaded frame against active cue and triggers treat toss if criteria are met |
| `POST` | `/predict` | Ad-hoc single frame vision inference returning `VisionPayload` |
| `POST` | `/stream/evaluate` | Evaluates a video file or live stream for $N$ frames and benchmarks latency/FPS |

---

## External Package Dependencies

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `fastapi` | $\ge 0.165.0$ | Asynchronous REST API framework |
| `uvicorn[standard]` | $\ge 0.54.0$ | ASGI server with hot-reloading |
| `ultralytics` | $\ge 8.4.218$ | YOLOv8 object detection runtime |
| `opencv-python` | $\ge 5.1.0.95$ | Video capture, frame decoding, drawing, video writing |
| `numpy` | $\ge 2.5.5$ | Numerical and array computations for vision coordinates |
| `scikit-learn` | $\ge 1.9.1$ | K-Means clustering for coat color classification |
| `pydantic` | $\ge 2.14.4$ | Data validation and schema serialization |
| `python-dotenv` | $\ge 1.2.4$ | Environment variable loading from `.env` |
| `python-multipart`| $\ge 0.0.34$ | Multipart form-data handling for image upload endpoints |
