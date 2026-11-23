import os

from dotenv import load_dotenv

load_dotenv()


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


APP_ENV = os.getenv("APP_ENV", "development").lower()

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    if APP_ENV == "production":
        raise RuntimeError("SECRET_KEY environment variable is not set. A secure SECRET_KEY is required in production.")
    SECRET_KEY = "dev-insecure-secret-key-change-in-production-min32bytes"

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./proctor.db")

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:5500,http://localhost:5500",
    ).split(",")
    if origin.strip()
]

ACCESS_TOKEN_EXPIRE_MINUTES = int(
    os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
)

REFRESH_TOKEN_EXPIRE_DAYS = int(
    os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7")
)

MAX_FRAME_BYTES = int(os.getenv("MAX_FRAME_BYTES", str(2 * 1024 * 1024)))

APP_NAME = os.getenv("APP_NAME", "Online Cheating Preventer API")


# --- Detector tuning (starting values; validate on a real webcam, see README) -----------
def _f(name, default): return float(os.getenv(name, str(default)))
def _i(name, default): return int(os.getenv(name, str(default)))

YAW_THRESHOLD_DEG = _f("YAW_THRESHOLD_DEG", 18.0)
LOOK_UP_THRESHOLD_DEG = _f("LOOK_UP_THRESHOLD_DEG", 25.0)
LOOK_DOWN_THRESHOLD_DEG = _f("LOOK_DOWN_THRESHOLD_DEG", 25.0)
DIRECTION_ENTER_FRAMES = _i("DIRECTION_ENTER_FRAMES", 3)
DIRECTION_CLEAR_FRAMES = _i("DIRECTION_CLEAR_FRAMES", 3)
CALIBRATION_SAMPLES = _i("CALIBRATION_SAMPLES", 20)
CALIBRATION_TIMEOUT_SECONDS = _f("CALIBRATION_TIMEOUT_SECONDS", 20.0)
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
