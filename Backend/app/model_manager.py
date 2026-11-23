
import logging
import os
from threading import Lock

from . import config

logger = logging.getLogger("proctor.models")


def _load_yolo_safely(model_path: str):
    """Load a trusted Ultralytics checkpoint with explicit safe globals."""

    import torch
    from ultralytics import YOLO

    from torch.nn import (
        Upsample,
        ModuleList,
        Sequential,
        BatchNorm2d,
        MaxPool2d,
        SiLU,
        Conv2d,
    )

    from ultralytics.nn.modules.conv import Conv, Concat
    from ultralytics.nn.modules.block import (
        DFL,
        C2f,
        Bottleneck,
        SPPF,
    )
    from ultralytics.nn.modules.head import Detect
    from ultralytics.nn.tasks import DetectionModel

    allowed_classes = [
        Upsample,
        Conv,
        ModuleList,
        Concat,
        DFL,
        Sequential,
        BatchNorm2d,
        C2f,
        MaxPool2d,
        SiLU,
        Bottleneck,
        Detect,
        SPPF,
        Conv2d,
        DetectionModel,
    ]

    with torch.serialization.safe_globals(allowed_classes):
        return YOLO(model_path)


class ModelManager:
    """Lazy-loads one YOLO model per worker process.

    A missing or invalid weights file disables phone detection
    without crashing the rest of the monitoring application.
    """

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or config.YOLO_MODEL_PATH
        self._model = None
        self._load_lock = Lock()
        self._load_failed = False
        self.load_error: str | None = None
        self.inference_lock = Lock()

    def get_yolo(self):
        if self._model is None and not self._load_failed:
            with self._load_lock:
                if self._model is None and not self._load_failed:
                    try:
                        if not os.path.isfile(self.model_path):
                            raise FileNotFoundError(
                                f"YOLO weights not found: {self.model_path}"
                            )

                        self._model = _load_yolo_safely(self.model_path)

                        logger.info(
                            "YOLO phone detection model loaded successfully: %s",
                            self.model_path,
                        )

                    except Exception as e:
                        self._load_failed = True
                        self.load_error = str(e)
                        logger.exception(
                            "Phone detection DISABLED - YOLO loading failed"
                        )

        return self._model

    def status(self) -> dict:
        return {
            "loaded": self._model is not None,
            "failed": self._load_failed,
            "error": self.load_error,
            "path": self.model_path,
        }

    def close(self):
        self._model = None


model_manager = ModelManager()
