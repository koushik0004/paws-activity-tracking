# PAWS Buddy — Modular Application Structure & Refactoring Documentation

## 1. Executive Summary & Thought Process

Previously, the PAWS Buddy codebase had two monolithic files at the root level:
- `pipeline.py` (392 lines): Mixed together data contracts, raw OpenCV/NumPy geometric calculations, K-Means clustering, hardware device negotiation, YOLO detection, RTMPose AP-10K keypoint inference, and temporal state hysteresis.
- `app.py` (304 lines): Combined FastAPI application instantiation, configuration, session state machine definition, route endpoint handlers, streaming evaluation benchmarking, and video writing logic.

### Problems Identified:
1. **Single-Responsibility Violation**: A bug fix or optimization in geometry/angle math required touching the same file as device hardware logic or model initialization.
2. **Poor LLM Uploadability**: When consulting or debugging with external LLM chat interfaces (such as Gemini Chat Web), dropping 400-line multi-concern files frequently hits token distractions or loses precision. Smaller, isolated, self-contained files (~50 to 250 lines) are much easier to review, test, and paste.
3. **Rigid Import Structure**: Testing individual endpoints or helper math required importing heavy ML frameworks (`ultralytics`, `torch`, `rtmlib`).

### Design Goals of the Splitting Pattern:
- **Logical Grouping**: Split based on concerns (Data contracts $\rightarrow$ Pure math/heuristics $\rightarrow$ Stateful model inference $\rightarrow$ HTTP routing $\rightarrow$ App bootstrapping).
- **Manageable File Sizes**: Keep individual files between 40 and 275 lines.
- **Zero Breaking Changes**: Provide root backward-compatible facades (`app.py` and `pipeline.py`) so that existing build tools, Docker containers, `Makefile` targets, and CI scripts continue to function seamlessly without modification.

---

## 2. Directory & Folder Structure

```
paws-buddy/
├── app.py                      # (47 lines) Backward-compatible facade for uvicorn & Makefile
├── pipeline.py                 # (31 lines) Backward-compatible facade for pipeline imports
├── src/
│   └── paws_buddy/             # Core Python package namespace
│       ├── __init__.py         # (36 lines) Unified public API package re-exports
│       ├── schemas.py          # (97 lines) Pure Pydantic v2 data models & AP-10K constants
│       ├── vision.py           # (130 lines) Pure computer vision, color clustering & posture math
│       ├── pipeline.py         # (235 lines) PawsVisionPipeline class & hardware device resolution
│       ├── routes.py           # (274 lines) FastAPI APIRouter endpoints & session evaluation
│       └── app.py              # (49 lines) Application factory create_app() & CLI entrypoint
└── docs/
    └── updated-app-structure.md # (This documentation file)
```

---

## 3. Function & Class Deconstruction Matrix

Here is the exact mapping of where functions, classes, and constants were relocated and the thought process behind each decision:

| Original Location | Symbol / Component | New Module | Thought Process & Rationale |
| :--- | :--- | :--- | :--- |
| `pipeline.py` | `AP10K_KEYPOINTS` | [src/paws_buddy/schemas.py](file:///Users/koushiksadhukhan/projects/paws-buddy/src/paws_buddy/schemas.py) | Pure data constant defining topology dictionary. Belongs alongside data contracts. |
| `pipeline.py` | `AppearanceSchema`, `PostureSchema`, `VisionPayload` | [src/paws_buddy/schemas.py](file:///Users/koushiksadhukhan/projects/paws-buddy/src/paws_buddy/schemas.py) | Vision output data contracts. Zero dependency on PyTorch or OpenCV; cleanly uploadable to Gemini web for schema review. |
| `app.py` | `CueRequest`, `SessionState`, `CueEvaluationResponse`, `PostureSegment`, `StreamBenchmarkReport` | [src/paws_buddy/schemas.py](file:///Users/koushiksadhukhan/projects/paws-buddy/src/paws_buddy/schemas.py) | Session state and HTTP response schemas. Grouped into `schemas.py` to establish a single source of truth for all data models. |
| `pipeline.py` | `_check_framing_margin` $\rightarrow$ `check_framing_margin` | [src/paws_buddy/vision.py](file:///Users/koushiksadhukhan/projects/paws-buddy/src/paws_buddy/vision.py) | Pure geometric boundary calculation. Extracted from the pipeline class so it can be tested independently with mock bounding boxes. |
| `pipeline.py` | `_extract_dominant_color` $\rightarrow$ `extract_dominant_color` | [src/paws_buddy/vision.py](file:///Users/koushiksadhukhan/projects/paws-buddy/src/paws_buddy/vision.py) | OpenCV + K-Means color clustering. Isolated as a pure utility function. |
| `pipeline.py` | `_joint_angle` $\rightarrow$ `joint_angle` | [src/paws_buddy/vision.py](file:///Users/koushiksadhukhan/projects/paws-buddy/src/paws_buddy/vision.py) | Vector cosine angle math using NumPy. Fully testable without running inference models. |
| `pipeline.py` | `_classify_ap10k_posture` $\rightarrow$ `classify_ap10k_posture` | [src/paws_buddy/vision.py](file:///Users/koushiksadhukhan/projects/paws-buddy/src/paws_buddy/vision.py) | AP-10K keypoint elevation and angle classification heuristic. Contains posture rules (SITTING, STANDING, DOWN, PAW_LIFT, CROUCHING). Easy to paste into Gemini chat for refining canine biomechanics heuristics. |
| `pipeline.py` | `resolve_device` | [src/paws_buddy/pipeline.py](file:///Users/koushiksadhukhan/projects/paws-buddy/src/paws_buddy/pipeline.py) | Hardware resolution logic (Metal Performance Shaders `mps` $\rightarrow$ CUDA $\rightarrow$ CPU). Placed with pipeline execution. |
| `pipeline.py` | `PawsVisionPipeline` | [src/paws_buddy/pipeline.py](file:///Users/koushiksadhukhan/projects/paws-buddy/src/paws_buddy/pipeline.py) | Stateful pipeline manager orchestrating YOLO detection, RTMPose keypoint extraction, temporal velocity calculation, hysteresis pose voting, and hold gating. |
| `app.py` | `current_session`, `_now_ms()`, `/health`, `/reset`, `/session/cue`, `/session/status`, `/session/evaluate_frame`, `/predict`, `/stream/evaluate` | [src/paws_buddy/routes.py](file:///Users/koushiksadhukhan/projects/paws-buddy/src/paws_buddy/routes.py) | FastAPI endpoint handlers and session evaluation logic packaged into an `APIRouter`. Includes dependency injection for `vision_engine`. |
| `app.py` | `create_app()`, `app`, `main()` | [src/paws_buddy/app.py](file:///Users/koushiksadhukhan/projects/paws-buddy/src/paws_buddy/app.py) | FastAPI application factory, middleware configuration, router mounting, and CLI executable entrypoint. |
| `app.py` (root) | Root re-exports | [app.py](file:///Users/koushiksadhukhan/projects/paws-buddy/app.py) | Backward compatibility shim re-exporting `app`, `current_session`, and routes for `uvicorn app:app` and `docker compose`. |
| `pipeline.py` (root) | Root re-exports | [pipeline.py](file:///Users/koushiksadhukhan/projects/paws-buddy/pipeline.py) | Backward compatibility shim re-exporting `PawsVisionPipeline`, `VisionPayload`, and schemas for existing scripts. |

---

## 4. Individual File Responsibilities

### [src/paws_buddy/schemas.py](file:///Users/koushiksadhukhan/projects/paws-buddy/src/paws_buddy/schemas.py) (97 lines)
- **Role:** Pure data contract layer.
- **Dependencies:** `pydantic`, `typing`.
- **Upload Value:** Lightweight and zero-dependency on ML libraries; ideal for sharing with Gemini chat to generate client SDKs, frontend TypeScript types, or documentation.

### [src/paws_buddy/vision.py](file:///Users/koushiksadhukhan/projects/paws-buddy/src/paws_buddy/vision.py) (130 lines)
- **Role:** Algorithmic and geometric computation.
- **Dependencies:** `cv2`, `numpy`, `scikit-learn`.
- **Upload Value:** Contains all posture thresholds (e.g. `forearm_lift > 0.14`, joint angles for sitting vs crouching). When posture classification needs tuning, this single file can be uploaded to Gemini for rapid iteration.

### [src/paws_buddy/pipeline.py](file:///Users/koushiksadhukhan/projects/paws-buddy/src/paws_buddy/pipeline.py) (235 lines)
- **Role:** Stateful model execution and temporal smoothing.
- **Dependencies:** `torch`, `ultralytics`, `rtmlib`, `schemas`, `vision`.
- **Upload Value:** Focuses exclusively on model weights, bounding box selection, velocity deadbands, and consecutive frame hysteresis without clutter from HTTP or data definitions.

### [src/paws_buddy/routes.py](file:///Users/koushiksadhukhan/projects/paws-buddy/src/paws_buddy/routes.py) (274 lines)
- **Role:** HTTP API transport & training session controller.
- **Dependencies:** `fastapi`, `schemas`, `pipeline`.
- **Upload Value:** Contains route declarations and video stream benchmarking. Decoupled from application initialization.

### [src/paws_buddy/app.py](file:///Users/koushiksadhukhan/projects/paws-buddy/src/paws_buddy/app.py) (49 lines)
- **Role:** Factory and lifecycle wiring.
- **Dependencies:** `fastapi`, `dotenv`, `routes`, `pipeline`.
- **Upload Value:** Minimal bootstrap file configuring the application instance and CLI entrypoint.

---

## 5. Backward Compatibility & Operational Verification

The modular structure preserves 100% backward compatibility:
1. `uv run uvicorn app:app --reload` continues to work seamlessly because root `app.py` re-exports `app`.
2. Existing Docker Compose (`docker-compose.yml`) configurations referencing `app:app` remain unaffected.
3. `make run`, `make docker-up`, and `make docker-build` execute without needing updates.
4. Python imports such as `from pipeline import PawsVisionPipeline` or `from app import app` work identical to prior releases.
