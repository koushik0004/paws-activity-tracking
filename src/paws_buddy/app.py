import os
from dotenv import load_dotenv
from fastapi import FastAPI

from paws_buddy.pipeline import PawsVisionPipeline
from paws_buddy.routes import router, init_vision_engine


def create_app() -> FastAPI:
    """Application factory for PAWS Buddy FastAPI microservice."""
    load_dotenv()

    device = os.getenv("DEVICE") or None  # Auto-resolves to "mps" on Mac M-series, "cuda", or "cpu"

    app = FastAPI(
        title="PAWS Buddy Treat Toss — Local Vision Inference API",
        version="0.2.0",
        description="Local FastAPI service running on Edge/Local inference accelerators.",
    )

    vision_engine = PawsVisionPipeline(
        detector_weights="yolov8n.pt",
        target_classes=[16, 0],
        framing_margin_pct=0.10,
        velocity_deadband_px_s=120.0,
        hold_gate_ms=300,
        fps=25.0,
        device=device,
    )

    init_vision_engine(vision_engine)
    app.include_router(router)
    app.state.vision_engine = vision_engine

    return app


app = create_app()


def main():
    """CLI entrypoint configured in pyproject.toml."""
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("paws_buddy.app:app", host="127.0.0.1", port=port, reload=True)


if __name__ == "__main__":
    main()
