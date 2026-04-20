import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file
# Defaults to looking for .env in current or parent directory
load_dotenv()

logger = logging.getLogger(__name__)

# Project Root Directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# --- Runtime Environment Control (local / colab) ---
ENV_MODE = os.getenv("ENV_MODE", "local").lower()

# --- Storage Path Retrieval Logic ---
if ENV_MODE == "colab":
    try:
        from google.colab import drive  # type: ignore # noqa: E402

        drive.mount("/content/drive")
    except ImportError:
        logger.warning("google.colab module not found, skipping Drive mount.")

    # Google Drive mounting paths
    ASSETS_DIR = os.getenv(
        "COLAB_ASSETS_DIR", "/content/drive/MyDrive/gpt-sovits/assets"
    )
    MODELS_DIR = os.getenv(
        "COLAB_MODELS_DIR", "/content/drive/MyDrive/gpt-sovits/models"
    )
else:
    # Local development paths
    ASSETS_DIR = os.getenv("LOCAL_ASSETS_DIR", str(PROJECT_ROOT / "assets"))
    MODELS_DIR = os.getenv("LOCAL_MODELS_DIR", str(PROJECT_ROOT / "models"))
