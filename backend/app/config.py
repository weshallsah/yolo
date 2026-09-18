import os
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    app_name: str = "Object Detection API"
    host: str = "0.0.0.0"
    port: int = 9000
    cors_origins: list[str] = field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]
    )
    yolo_weights_path: str = "yolov8n.pt"
    model_task: str = "detect"
    detection_confidence_threshold: float = 0.2


@lru_cache
def get_settings() -> Settings:
    port = int(os.environ.get("APP_PORT", 9000))
    weights_path = os.environ.get("APP_YOLO_WEIGHTS_PATH", "yolov8n.pt")
    model_task = os.environ.get("APP_MODEL_TASK", "detect").strip().lower()
    if model_task not in ("detect", "classify"):
        raise ValueError(f"APP_MODEL_TASK must be 'detect' or 'classify', got {model_task!r}")
    threshold = float(os.environ.get("APP_DETECTION_CONFIDENCE_THRESHOLD", 0.2))
    extra_origin = os.environ.get("APP_CORS_ORIGIN")

    cors_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    if extra_origin:
        cors_origins.append(extra_origin)

    return Settings(
        port=port,
        yolo_weights_path=weights_path,
        model_task=model_task,
        detection_confidence_threshold=threshold,
        cors_origins=cors_origins,
    )
