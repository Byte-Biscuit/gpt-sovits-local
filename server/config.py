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

# --- Runtime Environment Auto-Detection ---
# 1. First priority: Environment variable from .env or shell (ENV_MODE)
# 2. Second priority: Auto-detect Colab or WSL/Local
ENV_MODE = os.getenv("ENV_MODE", "").lower()

if not ENV_MODE:
    # Check for Colab specific environment variables or markers
    if os.path.exists("/content") and "COLAB_GPU" in os.environ:
        ENV_MODE = "colab"
    else:
        # Default to local/wsl
        ENV_MODE = "local"

logger.info(f"Runtime Environment Mode: {ENV_MODE.upper()}")

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

# 在Colab环境中使用该参数，主要用于避免训练时的Drive I/O导致的极速降低问题
# 在需要时直接读取此目录路径
TMP_MODELS_DIR = "/content/tmp_models"

# GPT-SoVITS Pretrained Models Directory
# We use the original project path as default because it is now mapped via symlink in Colab
# This ensures maximum compatibility with official code while using Drive storage.
PRETRAINED_DIR = str(PROJECT_ROOT / "GPT_SoVITS" / "pretrained_models")


# Training intermediate products and experimental logs
# Redefined: These are now stored within the speaker's folder in ASSETS_DIR for better organization.
# Usage example: os.path.join(get_speaker_logs_dir("wangliqun"), "train_steps.log")
def get_speaker_logs_dir(speaker_name: str) -> str:
    logs_path = os.path.join(ASSETS_DIR, speaker_name, "logs")
    os.makedirs(logs_path, exist_ok=True)
    return logs_path


def is_colab() -> bool:
    return ENV_MODE == "colab"
