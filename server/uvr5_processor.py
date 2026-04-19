"""
UVR5 人声分离处理模块

职责：
    对切片后的音频执行人声/伴奏分离，输出用于训练的纯净干声。

目录约定：
    models/uvr5/                             ← 模型权重文件存放目录
    assets/{speaker}/sliced/          ← 输入（切片结果）
    assets/{speaker}/vocals/          ← 输出干声（送入 ASR 打标）
    assets/{speaker}/instrumental/    ← 输出伴奏（训练时可丢弃）

推荐处理链（单人音色克隆）：
    1. HP5_only_main_vocal  → 分离主唱干声
    2. VR-DeEchoNormal      → 去混响（录音环境干净可跳过）

模型类型与对应类：
    HP2_all_vocals / HP5_only_main_vocal       → AudioPre（vr.py）
    VR-DeEchoNormal / VR-DeEchoAggressive     → AudioPreDeEcho（vr.py）
    bs_roformer_* / *mel_band_roformer*        → Roformer_Loader（bsroformer.py）
"""

import logging
import os
import sys

import torch

# 将项目根目录加入 sys.path，确保 tools.uvr5 可被导入
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# tools/uvr5 目录也需在路径中，因为 vr.py 内部使用相对 import
_uvr5_dir = os.path.join(_project_root, "tools", "uvr5")
if _uvr5_dir not in sys.path:
    sys.path.insert(0, _uvr5_dir)

from server.logger import setup_logging  # noqa: E402
from tools.uvr5.bsroformer import Roformer_Loader  # noqa: E402
from tools.uvr5.vr import AudioPre, AudioPreDeEcho  # noqa: E402

logger = logging.getLogger("server.uvr5_processor")

# 模型权重统一存放在项目根目录下的 models/uvr5/
MODELS_DIR = os.path.join(_project_root, "models", "uvr5")

# 各模型文件名（与 models/uvr5/ 目录中的文件名对应）
MODEL_HP5 = "HP5_only_main_vocal.pth"  # 只保留主唱，训练数据首选
MODEL_HP2 = "HP2_all_vocals.pth"  # 保留所有人声（含和声）
MODEL_DEECHO_NORMAL = "VR-DeEchoNormal.pth"  # 温和去混响
MODEL_DEECHO_AGGRESSIVE = "VR-DeEchoAggressive.pth"  # 强力去混响
# Roformer 系列（高质量，需 GPU，.ckpt 与同名 .yaml 必须成对存放）
MODEL_BS_ROFORMER = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"  # 综合质量最高
MODEL_MEL_ROFORMER = "kim_mel_band_roformer.ckpt"  # 人声细节最优，TTS 首选


def _detect_device() -> tuple[str, bool]:
    """
    自动选择推理设备。

    Returns
    -------
    device : str
        "cuda" 或 "cpu"
    is_half : bool
        GPU 时启用半精度（FP16）以节省显存；CPU 时强制 False
    """
    if torch.cuda.is_available():
        return "cuda", True
    return "cpu", False


class UVR5Processor:
    """
    封装 UVR5 人声分离流程。

    模型按需懒加载（首次调用时才载入显存），避免不必要的显存占用。

    Parameters
    ----------
    models_dir : str
        模型权重目录，默认 models/uvr5/
    agg : int
        人声提取激进程度 0~20，越大提取越激进但可能损失音质，默认 10
    output_format : str
        输出音频格式，"wav" 或 "flac"，默认 "wav"
    device : str | None
        推理设备，None 表示自动检测
    is_half : bool | None
        是否使用半精度，None 表示自动（GPU 时 True，CPU 时 False）
    """

    def __init__(
        self,
        models_dir: str = MODELS_DIR,
        agg: int = 10,
        output_format: str = "wav",
        device: str | None = None,
        is_half: bool | None = None,
    ):
        self.models_dir = os.path.abspath(models_dir)
        self.agg = agg
        self.output_format = output_format

        # 自动检测或使用指定设备
        auto_device, auto_half = _detect_device()
        self.device = device if device is not None else auto_device
        self.is_half = is_half if is_half is not None else auto_half

        logger.info(
            "UVR5Processor 初始化 | 设备=%s | 半精度=%s | 模型目录=%s",
            self.device,
            self.is_half,
            self.models_dir,
        )

        # 懒加载缓存：避免重复加载同一模型
        self._model_cache: dict[str, AudioPre | AudioPreDeEcho | Roformer_Loader] = {}

    # ------------------------------------------------------------------
    # 私有方法
    # ------------------------------------------------------------------

    def _model_path(self, filename: str) -> str:
        """拼接模型完整路径，并检查文件是否存在。"""
        path = os.path.join(self.models_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"模型文件不存在: {path}\n"
                f"请先运行 python -m server.models_loader 下载模型权重。"
            )
        return path

    def _load_model(
        self, model_filename: str
    ) -> AudioPre | AudioPreDeEcho | Roformer_Loader:
        """
        懒加载模型：已加载的模型从缓存中复用，避免重复占用显存。

        模型类型判断优先级（依据文件名）：
          1. 含 'roformer'              → Roformer_Loader（bs_roformer / mel_band_roformer）
          2. 含 'DeEcho' 或 'DeReverb' → AudioPreDeEcho（去混响/回声）
          3. 其余                       → AudioPre（标准人声/伴奏分离）
        """
        if model_filename in self._model_cache:
            return self._model_cache[model_filename]

        path = self._model_path(model_filename)
        logger.info("正在加载模型: %s", model_filename)

        name_lower = model_filename.lower()
        if "roformer" in name_lower:
            # Roformer 系列需要同名 .yaml 配置文件与 .ckpt 文件成对存放
            config_path = os.path.splitext(path)[0] + ".yaml"
            model = Roformer_Loader(
                model_path=path,
                config_path=config_path,
                device=self.device,
                is_half=self.is_half,
            )
        elif "deecho" in name_lower or "dereverb" in name_lower:
            model = AudioPreDeEcho(
                agg=self.agg,
                model_path=path,
                device=self.device,
                is_half=self.is_half,
            )
        else:
            model = AudioPre(
                agg=self.agg,
                model_path=path,
                device=self.device,
                is_half=self.is_half,
            )

        self._model_cache[model_filename] = model
        logger.info("模型加载完成: %s", model_filename)
        return model

    def _run_separation(
        self,
        model_filename: str,
        input_dir: str,
        vocal_dir: str,
        ins_dir: str,
    ) -> list[str]:
        """
        对 input_dir 下所有 WAV 文件执行分离，返回成功处理的文件列表。

        ins_dir 传 None 可跳过伴奏输出以节省磁盘空间。
        """
        os.makedirs(vocal_dir, exist_ok=True)
        os.makedirs(ins_dir, exist_ok=True)

        model = self._load_model(model_filename)

        wav_files = sorted(
            f for f in os.listdir(input_dir) if f.lower().endswith(".wav")
        )
        if not wav_files:
            logger.warning("输入目录中未找到 WAV 文件: %s", input_dir)
            return []

        processed = []
        for wav in wav_files:
            inp = os.path.join(input_dir, wav)
            logger.info("处理: %s", inp)
            try:
                # 三个模型类的 _path_audio_ 参数名和顺序不同，必须分别调用：
                #   AudioPre:      (music_file, ins_root,   vocal_root, format)
                #   AudioPreDeEcho:(music_file, vocal_root, ins_root,   format)  ← vocal/ins 顺序相反
                #   Roformer_Loader:(input,    others_root, vocal_root, format)  ← 参数名不同
                if isinstance(model, Roformer_Loader):
                    model._path_audio_(
                        input=inp,
                        others_root=ins_dir,
                        vocal_root=vocal_dir,
                        format=self.output_format,
                    )
                elif isinstance(model, AudioPreDeEcho):
                    model._path_audio_(
                        music_file=inp,
                        vocal_root=vocal_dir,
                        ins_root=ins_dir,
                        format=self.output_format,
                    )
                else:
                    model._path_audio_(
                        music_file=inp,
                        ins_root=ins_dir,
                        vocal_root=vocal_dir,
                        format=self.output_format,
                    )
                processed.append(inp)
            except Exception as exc:
                logger.error("分离失败: %s | 原因: %s", inp, exc)

        logger.info("分离完成，共处理 %d 个文件 -> %s", len(processed), vocal_dir)
        return processed

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def separate_vocals(
        self,
        speaker_dir: str,
        input_subdir: str = "sliced",
        vocal_subdir: str = "vocals",
        ins_subdir: str = "instrumental",
        model_filename: str = MODEL_BS_ROFORMER,
    ) -> list[str]:
        """
        对音色目录执行人声/伴奏分离。

        Parameters
        ----------
        speaker_dir : str
            音色根目录，例如 assets/wangliqun
        input_subdir : str
            输入子目录（切片结果），默认 sliced
        vocal_subdir : str
            干声输出子目录，默认 vocals
        ins_subdir : str
            伴奏输出子目录，默认 instrumental
        model_filename : str
            使用的模型文件名，默认 model_bs_roformer_ep_317_sdr_12.9755.ckpt

        Returns
        -------
        list[str]
            成功处理的输入文件路径列表
        """
        speaker_dir = os.path.abspath(speaker_dir)
        return self._run_separation(
            model_filename=model_filename,
            input_dir=os.path.join(speaker_dir, input_subdir),
            vocal_dir=os.path.join(speaker_dir, vocal_subdir),
            ins_dir=os.path.join(speaker_dir, ins_subdir),
        )

    def separate_from_dir(
        self,
        input_dir: str,
        vocal_dir: str,
        ins_dir: str,
        model_filename: str = MODEL_BS_ROFORMER,
    ) -> list[str]:
        """
        直接指定输入/输出目录执行人声分离，不依赖 speaker_dir 约定。

        Parameters
        ----------
        input_dir : str
            切片 WAV 文件所在目录（完整路径）
        vocal_dir : str
            干声输出目录（完整路径，不存在时自动创建）
        ins_dir : str
            伴奏输出目录（完整路径，不存在时自动创建）
        model_filename : str
            使用的模型文件名，默认 MODEL_BS_ROFORMER

        Returns
        -------
        list[str]
            成功处理的输入文件路径列表
        """
        return self._run_separation(
            model_filename=model_filename,
            input_dir=os.path.abspath(input_dir),
            vocal_dir=os.path.abspath(vocal_dir),
            ins_dir=os.path.abspath(ins_dir),
        )

    def remove_echo(
        self,
        speaker_dir: str,
        input_subdir: str = "vocals",
        vocal_subdir: str = "vocals_clean",
        ins_subdir: str = "vocals_echo",
        aggressive: bool = False,
    ) -> list[str]:
        """
        对干声目录执行去混响/回声处理（可选步骤）。

        Parameters
        ----------
        speaker_dir : str
            音色根目录
        input_subdir : str
            输入子目录（人声分离结果），默认 vocals
        vocal_subdir : str
            去混响后输出子目录，默认 vocals_clean
        ins_subdir : str
            混响成分输出子目录，默认 vocals_echo
        aggressive : bool
            True 使用强力去混响，False 使用温和模式，默认 False
        """
        model_filename = MODEL_DEECHO_AGGRESSIVE if aggressive else MODEL_DEECHO_NORMAL
        speaker_dir = os.path.abspath(speaker_dir)
        return self._run_separation(
            model_filename=model_filename,
            input_dir=os.path.join(speaker_dir, input_subdir),
            vocal_dir=os.path.join(speaker_dir, vocal_subdir),
            ins_dir=os.path.join(speaker_dir, ins_subdir),
        )

    def release_models(self) -> None:
        """释放所有已加载模型的显存，在处理完成后调用。"""
        self._model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("模型缓存已释放")


if __name__ == "__main__":
    setup_logging()

    # ---- 在此修改要处理的音色目录和参数 ----
    SPEAKER_DIR = "assets/wangliqun"
    AGG = 10  # 人声提取激进程度 0~20
    REMOVE_ECHO = False  # 是否执行去混响（录音干净可关闭）
    ECHO_MODE = False  # True=强力去混响, False=温和去混响
    # ----------------------------------------

    processor = UVR5Processor(agg=AGG)

    # 第一步：人声/伴奏分离（使用 bs_roformer，models/uvr5/ 下需有对应 .ckpt）
    processor.separate_vocals(SPEAKER_DIR)

    # 第二步：去混响（可选）
    if REMOVE_ECHO:
        processor.remove_echo(SPEAKER_DIR, aggressive=ECHO_MODE)

    # 释放显存
    processor.release_models()
