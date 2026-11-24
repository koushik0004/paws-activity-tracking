# PAWS Buddy 🐾

> Intelligent Edge Computer Vision Inference and Automated Pet Training Microservice.

PAWS Buddy is a real-time computer vision service designed to assist in automated pet training and treat dispensing. It runs locally or inside containers using FastAPI, YOLOv8, and OpenCV, supporting hardware acceleration (Apple Silicon MPS, NVIDIA CUDA, or CPU fallback).

---

## Key Capabilities

- **Real-Time Detection & Proxy Filtering**: Detects dogs (`cls=16`) and human proxy testing classes (`cls=0`) with YOLOv8.
- **Safe Framing Margins**: Enforces a configurable bounding safety margin (default 10% edge padding) to guarantee that pets are fully in frame before rewarding.
- **Coat Color Extraction**: Extracts dominant coat RGB and Hex values using K-Means clustering on the detected subject's region of interest (ROI).
- **Kinematic Stability & Hold Gating**: Tracks centroid velocity over a smoothed rolling window (`smooth_window=5`). Distinguishes between `MOVING` and `SITTING` postures, tracking continuous hold duration ($\ge 300\text{ms}$).
- **Interactive Training State Machine**:
  - `IDLE` $\rightarrow$ `ACTIVE` $\rightarrow$ `MARKED` (treat triggered) or `EXPIRED` (timeout).
  - Enforces a strict **one-reward-per-cue** guarantee.
- **Flexible Deployment**: Native local execution via `uv` or containerized execution via Docker Compose with live source code volume mounting.

---

## Architecture Overview

```
                      +---------------------------------------+
                      |     Client / Video Stream / Web UI    |
                      +-------------------+-------------------+
                                          |
                      +-------------------v-------------------+
                      |      FastAPI Server (app.py)          |
                      +-------------------+-------------------+
                                          |
                      +-------------------v-------------------+
                      |  PawsVisionPipeline (pipeline.py)     |
                      |  - YOLOv8 Object Detection            |
                      |  - 10% Framing Margin Check           |
                      |  - K-Means Dominant Color             |
                      |  - Centroid Velocity & Hold Tracking  |
                      +-------------------+-------------------+
                                          |
                      +-------------------v-------------------+
                      |    Session State Machine (CTRL-01)    |
                      |    ACTIVE -> Hold >= 300ms -> MARKED  |
                      +-------------------+-------------------+
                                          |
                                          v
                      [Trigger ESP32 Treat Tosser Hardware]
```

---

## Prerequisites

- **Python**: `>= 3.12`
- **Package Manager**: [`uv`](https://docs.astral.sh/uv/) (recommended)
- **Container Runtime** (optional): [Docker Desktop](https://www.docker.com/) & Docker Compose
- **Model Weights**: `yolov8n.pt` (included at repo root)
- **FFmpeg / OpenCV dependencies**: If running on headless Linux, install `libgl1` and `libglib2.0-0`.

---

## Configuration & Environment Variables

Create or edit a `.env` file at the repository root to configure runtime behaviors:

```env
# Server Port (Environment variable has highest priority)
PORT=8000

# Evaluation Mode: true uses TEST_VIDEO_PATH, false connects to LIVE_CAMERA_INDEX
EVAL_MODE=true

# Inference Device: "mps" (Mac M-series), "cuda" (NVIDIA), or "cpu"
DEVICE=mps

# Video Source for Evaluation
TEST_VIDEO_PATH=data/dog-1.mp4

# Live Camera Device Index (used when EVAL_MODE=false)
LIVE_CAMERA_INDEX=0
```

### Port Precedence Logic
Port resolution adheres to the following priority:
1. **Shell Environment Variable**: `PORT=9000 make run` *(Highest priority)*
2. **`.env` File Setting**: `PORT=...` defined in `.env`
3. **Makefile Command-Line Argument**: `make run PORT=9000`
4. **Default Fallback**: `8000`

---

## How to Run the Application

You can run the application either directly on your host machine or within a Docker container.

### Method 1: Local Execution via `Makefile` (Recommended)

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Run the API server with auto-reload**:
   ```bash
   # Run with default port (8000) or PORT from .env
   make run

   # Override port via environment variable (highest priority)
   PORT=8080 make run

   # Override port via make argument
   make run PORT=9000
   ```

3. **Start a public tunnel** (Optional, requires `cloudflared`):
   ```bash
   make tunnel
   ```

---

### Method 2: Local Execution via Direct `uv` Command

```bash
uv run uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

---

### Method 3: Containerized Execution via Docker Compose

Docker Compose mounts your workspace live into `/app`, allowing edits made on your host machine to instantly reload inside the running container without rebuilding the image.

1. **Start the container in detached mode**:
   ```bash
   make docker-up
   # or: docker compose up -d
   ```

2. **View live container logs**:
   ```bash
   make docker-logs
   # or: docker compose logs -f
   ```

3. **Stop the container**:
   ```bash
   make docker-down
   # or: docker compose down
   ```

4. **Rebuild container image** (when dependencies or Dockerfile change):
   ```bash
   make docker-build
   # or: docker compose build
   ```

---

## API Endpoints & Usage

Once running, interactive OpenAPI Swagger documentation is available at:
👉 **`http://localhost:8000/docs`** (replace `8000` with your configured port).

### 1. Health Check
```bash
curl -X GET http://localhost:8000/health
```
**Response:**
```json
{
  "status": "healthy",
  "device": "mps",
  "eval_mode": true,
  "frame_counter": 0,
  "hold_gate_ms": 300
}
```

### 2. Issue a Training Cue
Starts a session window (e.g. 5 seconds) awaiting the pet to perform a posture:
```bash
curl -X POST http://localhost:8000/session/cue \
  -H "Content-Type: application/json" \
  -d '{"command_pose": "SITTING", "window_ms": 5000}'
```

### 3. Evaluate Frame Against Cue
Uploads a frame from camera or stream. If the pet holds the command pose for $\ge 300\text{ms}$ within valid framing margins, `trigger_esp32_toss` becomes `true`:
```bash
curl -X POST http://localhost:8000/session/evaluate_frame \
  -F "file=@data/dog-1.mp4"
```

### 4. Direct Single-Frame Inference
Performs object detection, coat color extraction, and posture estimation on a single image:
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@data/test_frame.jpg"
```

### 5. Stream Benchmark Evaluation
Processes $N$ frames from the evaluation video or webcam, returns latency percentiles, effective FPS, and outputs an annotated debug video:
```bash
curl -X POST "http://localhost:8000/stream/evaluate?max_frames=50&save_annotated=true"
```
Annotated videos are saved in the `outputs/` directory (e.g. `outputs/local_stream_verified.mp4`).

---

## Project Structure

```
paws-buddy/
├── app.py                  # FastAPI application & session state machine routes
├── pipeline.py             # Vision pipeline (YOLOv8, framing, kinematics, color)
├── yolov8n.pt              # Pretrained YOLOv8 nano weights
├── Makefile                # Automation commands (make run, docker-up, etc.)
├── Dockerfile              # Container definition with uv & uvicorn
├── docker-compose.yml      # Service setup with hot-reload volume mounting
├── pyproject.toml          # Project metadata and dependencies managed by uv
├── .env                    # Runtime environment configuration
├── data/                   # Sample test videos (dog-1.mp4, huskey-1.mp4, etc.)
├── outputs/                # Evaluation benchmark videos and debug output
└── src/paws_buddy/         # Core package source directory
```

---

## Context & Agent Documentation

This repository includes standardized root context files for AI coding agents and developers:
- [GEMINI.md](file:///Users/koushiksadhukhan/projects/paws-buddy/GEMINI.md) — Agent steering instructions, persona, and rules.
- [project-map.md](file:///Users/koushiksadhukhan/projects/paws-buddy/project-map.md) — Directory hierarchy and entry points.
- [project-catalog.md](file:///Users/koushiksadhukhan/projects/paws-buddy/project-catalog.md) — Schemas, endpoints, and external dependencies index.
- [project-architecture.md](file:///Users/koushiksadhukhan/projects/paws-buddy/project-architecture.md) — System architecture and state transition diagrams.
- [skills.md](file:///Users/koushiksadhukhan/projects/paws-buddy/skills.md) — Developer and agent operational sub-skills.
