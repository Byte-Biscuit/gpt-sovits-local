import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
# 默认查找当前目录或父目录中的 .env
load_dotenv()

logger = logging.getLogger(__name__)

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# --- 运行环境控制 (local / colab) ---
ENV_MODE = os.getenv("ENV_MODE", "local").lower()

# --- 存储路径读取逻辑 ---
if ENV_MODE == "colab":
    # Google Drive 挂载路径
    ASSETS_DIR = os.getenv("COLAB_ASSETS_DIR", "/content/drive/MyDrive/gpt-sovits/assets")
    MODELS_DIR = os.getenv("COLAB_MODELS_DIR", "/content/drive/MyDrive/gpt-sovits/models")
else:
    # 本地开发路径
    ASSETS_DIR = os.getenv("LOCAL_ASSETS_DIR", str(PROJECT_ROOT / "assets"))
    MODELS_DIR = os.getenv("LOCAL_MODELS_DIR", str(PROJECT_ROOT / "models"))
