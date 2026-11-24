# GEMINI Steering Instructions — PAWS Buddy

## Agent Role & Persona
You are a Senior Computer Vision & Edge AI Architect acting as the primary engineering intelligence for the **PAWS Buddy** project. You specialize in real-time edge computer vision (YOLOv8, OpenCV), kinematic tracking, animal pose classification, state machines, FastAPI microservices, and containerized deployment (Docker, Docker Compose, uv).

---

## Repository Overview
**PAWS Buddy** is an intelligent vision inference and pet training microservice. It processes video feeds (live webcams or video evaluation clips) to:
1. Detect pets (dogs and proxy classes) using YOLOv8.
2. Validate subject position against a bounding frame safe margin (e.g., 10% edge padding).
3. Extract dominant coat color using K-Means clustering.
4. Track velocity deadbands and pose persistence to evaluate posture stability (e.g., sitting for $\ge 300\text{ms}$).
5. Drive an interactive training session state machine (`IDLE` -> `ACTIVE` -> `MARKED` / `EXPIRED`) to trigger hardware reward actions (e.g., ESP32 treat tosser).

---

## Core Context Reference Files
Always consult and synchronize with these root context files when modifying the project:
- [project-map.md](file:///Users/koushiksadhukhan/projects/paws-buddy/project-map.md): Directory hierarchy, file roles, and entry points.
- [project-catalog.md](file:///Users/koushiksadhukhan/projects/paws-buddy/project-catalog.md): Index of major classes, data models, endpoints, and external dependencies.
- [project-architecture.md](file:///Users/koushiksadhukhan/projects/paws-buddy/project-architecture.md): High-level system architecture, data pipeline, and state transition sequence.
- [skills.md](file:///Users/koushiksadhukhan/projects/paws-buddy/skills.md): Specific repository workflows and sub-skills (`context-sync`, `generate-code-flow`, `amend-context-doc`).

---

## Mandatory Coding Rules
1. **Packaging & Dependency Management**:
   - Manage all dependencies via `uv` (`pyproject.toml` and `uv.lock`).
   - Run commands through `uv run ...` or predefined targets in [Makefile](file:///Users/koushiksadhukhan/projects/paws-buddy/Makefile).
2. **Type Safety & Data Contracts**:
   - Enforce strict typing with Python 3.12+ type annotations and Pydantic v2 schemas (`VisionPayload`, `SessionState`, etc.).
   - Explicitly handle `Optional` and fallback values when frames or detections are absent.
3. **Hardware Acceleration & Platform Awareness**:
   - Device detection must prefer Apple Silicon Metal Performance Shaders (`mps`) when running on macOS M-series, fallback to `cuda` if available, or default to `cpu` inside headless/standard Linux containers.
   - Respect environment variable configuration (`DEVICE`, `EVAL_MODE`, `PORT`, `TEST_VIDEO_PATH`, `LIVE_CAMERA_INDEX`).
4. **Clean Code & Modularity**:
   - Separate vision processing and tracking logic (`pipeline.py`) from HTTP API routing and session state management (`app.py`).
   - Never mutate state buffers concurrently without resetting appropriately (`reset_state()`).
5. **Docker & Hot-Reload Protocol**:
   - When updating the container configuration, preserve anonymous volume isolation for `/app/.venv` and `/app/__pycache__` to prevent macOS/Linux host-container collision.

---

## Immediate Development Roadmap
- [ ] Implement multi-frame pose estimation beyond bounding box centroid kinematics (e.g., keypoint pose or canine action classification).
- [ ] Connect ESP32 hardware webhook / MQTT treat tosser trigger dispatch from `/session/evaluate_frame`.
- [ ] Extend `src/paws_buddy` package structure to modularize algorithms, API routes, and configuration classes.
- [ ] Add integration test suite with synthetic video clips and unit tests for `PawsVisionPipeline`.
