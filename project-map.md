# PAWS Buddy Project Map

## Directory Tree

```
paws-buddy/
├── .agents/
│   └── skills/                  # Repository skills and workflows
├── data/                        # Sample video datasets for testing & benchmarking
│   ├── dog-1.mp4
│   ├── huskey-1.mp4
│   ├── huskey-2.mp4
│   ├── spenial-1.mp4
│   └── spenial-2.mp4
├── outputs/                     # Generated benchmark MP4s and evaluation artifacts
│   └── local_stream_verified.mp4
├── src/
│   └── paws_buddy/              # Core installable Python package module
│       ├── __init__.py          # Package entry point & root export facade
│       ├── schemas.py           # Pydantic schemas (Vision & Session contracts)
│       ├── vision.py            # Geometric math, framing margin, color & AP-10K classifier
│       ├── pipeline.py          # PawsVisionPipeline class & hardware device resolution
│       ├── routes.py            # FastAPI APIRouter endpoints & session evaluation
│       └── app.py               # Application factory `create_app()` & CLI main entrypoint
├── .env                         # Environment variables (runtime overrides)
├── .gitignore                   # Git exclusion rules
├── app.py                       # Root backward-compatible entrypoint shim for uvicorn
├── pipeline.py                  # Root backward-compatible entrypoint shim for pipeline imports
├── yolov8n.pt                   # Pretrained YOLOv8 nano model weights
├── Dockerfile                   # Container specification with uv & uvicorn hot-reload
├── docker-compose.yml           # Compose service with live volume mounting
├── Makefile                     # Developer automation targets (run, docker-*, tunnel)
├── pyproject.toml               # Project metadata & uv dependencies
├── uv.lock                      # Locked dependency versions
├── GEMINI.md                    # Agent steering instructions & persona
├── project-map.md               # Directory layout and entry points (this file)
├── project-catalog.md           # Catalog of classes, schemas, endpoints & dependencies
├── project-architecture.md      # System architecture, data flow & state diagrams
└── skills.md                    # Operational developer sub-skills
```

---

## Core Components & Entry Points

| File / Directory | Role / Responsibility | Key Functions / Entry Points |
| :--- | :--- | :--- |
| [app.py](file:///Users/koushiksadhukhan/projects/paws-buddy/app.py) | Primary HTTP API service & session state management | `app` (FastAPI instance), `/session/*`, `/predict`, `/stream/evaluate` |
| [pipeline.py](file:///Users/koushiksadhukhan/projects/paws-buddy/pipeline.py) | Real-time CV inference, framing check & posture tracker | `PawsVisionPipeline`, `process_frame()`, `reset_state()` |
| [Makefile](file:///Users/koushiksadhukhan/projects/paws-buddy/Makefile) | CLI command automation and environment port precedence | `make run`, `make docker-up`, `make docker-down`, `make tunnel` |
| [docker-compose.yml](file:///Users/koushiksadhukhan/projects/paws-buddy/docker-compose.yml) | Docker service definition with volume mounting & reload | Service `web`, port forwarding `"${PORT:-8000}:${PORT:-8000}"` |
| [Dockerfile](file:///Users/koushiksadhukhan/projects/paws-buddy/Dockerfile) | Production/dev container image using `ghcr.io/astral-sh/uv` | UV project environment, OpenSSL/libGL install, uvicorn entrypoint |
| [pyproject.toml](file:///Users/koushiksadhukhan/projects/paws-buddy/pyproject.toml) | Python dependency specification and project metadata | Dependencies: `fastapi`, `ultralytics`, `opencv-python`, etc. |
| [src/paws_buddy/](file:///Users/koushiksadhukhan/projects/paws-buddy/src/paws_buddy) | Modular package source directory | Package initializer and future library modules |
| [data/](file:///Users/koushiksadhukhan/projects/paws-buddy/data) | Test video store for local evaluation mode | `dog-1.mp4`, `huskey-1.mp4`, `spenial-1.mp4`, etc. |
| [outputs/](file:///Users/koushiksadhukhan/projects/paws-buddy/outputs) | Output directory for annotated video evaluation runs | `local_stream_verified.mp4` |
