import logging
import os
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from server.config import MODELS_DIR, PROJECT_ROOT
from server.logger import setup_logging
from server.proxy import setup_proxy

logger = logging.getLogger("server.models_loader")


def download_gpt_sovits_models():
    """
    下载 v2ProPlus 微调训练所需的全部预训练模型。

    各文件用途：
        chinese-hubert-base/
            HuBERT 音频语义特征提取模型（特征提取 Step 2）
            输出每帧 768 维向量，是连接 GPT 和 SoVITS 的语义桥梁

        chinese-roberta-wwm-ext-large/
            中文 BERT 文本语义模型（特征提取 Step 1）
            为每个音素生成 1024 维上下文向量，让 GPT 理解文本含义

        sv/pretrained_eres2netv2w24s4ep4.ckpt
            ERes2NetV2 说话人验证模型（特征提取 Step 3，v2Pro 专有）
            提取说话人 d-vector，训练时约束音色稳定性

        v2Pro/s2Gv2ProPlus.pth
            SoVITS v2ProPlus 生成器预训练权重，两处用途：
            1. 特征提取 Step 4：VQ 量化器将 HuBERT 离散化为语义 token
            2. SoVITS 微调：作为生成器的初始权重（在此基础上学习目标音色）

        v2Pro/s2Dv2ProPlus.pth
            SoVITS v2ProPlus 判别器预训练权重
            SoVITS 微调时与生成器对抗训练，保证输出音质

        gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt
            GPT 预训练权重（5000小时训练，比 2kh 版本更强）
            GPT 微调的初始权重，在此基础上学习目标说话人的韵律节奏
    """
    logger.info("下载 GPT-SoVITS v2ProPlus 微调所需预训练模型...")
    snapshot_download(
        repo_id="lj1995/GPT-SoVITS",
        local_dir=os.path.join(MODELS_DIR, "pretrained"),
        allow_patterns=[
            # ── 特征提取模型 ──────────────────────────────────────────
            "chinese-hubert-base/*",  # Step2: HuBERT 音频语义特征
            "chinese-roberta-wwm-ext-large/*",  # Step1: BERT 文本语义特征
            "sv/*",  # Step3: 说话人验证（v2Pro 专有）
            # ── SoVITS v2ProPlus 预训练权重 ───────────────────────────
            "v2Pro/s2Gv2ProPlus.pth",  # 生成器（Step4 VQ量化 + 微调初始权重）
            "v2Pro/s2Dv2ProPlus.pth",  # 判别器（微调对抗训练用）
            "v2Pro/s2Gv2Pro.pth",
            "v2Pro/s2Dv2Pro.pth",
            # ── GPT 预训练权重 ────────────────────────────────────────
            "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        ],
    )


def download_g2pw_models():
    logger.info("Downloading G2PW Chinese polyphone alignment models...")
    # Used for improving Mandarin TTS polyphone disambiguation accuracy [3, 4]
    snapshot_download(
        repo_id="lj1995/G2PWModel",
        local_dir=os.path.join(MODELS_DIR, "text", "G2PWModel"),
    )


def download_uvr5_weights():
    logger.info("Downloading UVR5 traditional VR model weights...")
    # Used for extracting clean vocals from original audio [5, 6]
    uvr5_models = [
        "HP2_all_vocals.pth",
        "HP5_only_main_vocal.pth",
        "VR-DeEchoAggressive.pth",
        "VR-DeEchoNormal.pth",
        "Onnx_Aggressive_DeReverb.pth",
    ]
    for model in uvr5_models:
        hf_hub_download(
            repo_id="lj1995/UVR5",
            filename=model,
            local_dir=os.path.join(MODELS_DIR, "uvr5"),
        )


def download_roformer_weights():
    """
    Downloads BS-Roformer and Mel-Band-Roformer model weights and accompanying config files.

    Downloads uvr5_weights.zip from XXXXRT/GPT-SoVITS-Pretrained,
    then extracts Roformer-related files into models/uvr5/.
    Each model consists of a .ckpt (weighs) and a .yaml (config) file,
    filenames must match (excluding extension) and must contain 'roformer'.
    """
    logger.info("Downloading uvr5_weights.zip from XXXXRT/GPT-SoVITS-Pretrained...")
    zip_path = hf_hub_download(
        repo_id="XXXXRT/GPT-SoVITS-Pretrained",
        filename="uvr5_weights.zip",
        cache_dir=os.path.join(MODELS_DIR, "_cache"),
    )

    # Roformer files to extract (must be in .ckpt and .yaml pairs)
    roformer_targets = {
        "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        "model_bs_roformer_ep_368_sdr_12.9628.yaml",
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "model_bs_roformer_ep_317_sdr_12.9755.yaml",
        "kim_mel_band_roformer.ckpt",
        "kim_mel_band_roformer.yaml",
    }

    uvr5_dir = os.path.join(MODELS_DIR, "uvr5")
    os.makedirs(uvr5_dir, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for entry in zf.namelist():
            basename = os.path.basename(entry)
            if basename in roformer_targets:
                out_path = os.path.join(uvr5_dir, basename)
                with zf.open(entry) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                extracted.append(basename)
                logger.info("Extracted: %s", basename)

    if not extracted:
        logger.warning(
            "Roformer model files not found in archive, please place manually in models/uvr5/"
        )
    else:
        logger.info(
            "Roformer extraction complete, %d files -> %s", len(extracted), uvr5_dir
        )


def migrate_pretrained_dir() -> Path:
    """
    将 GPT_SoVITS/pretrained_models/ 迁移到 models/pretrained/，
    并在原位置创建符号链接，使原有代码无需任何修改即可继续运行。

    迁移后目录结构：
        models/pretrained/          ← 实际存储位置（纳入 models/ 统一管理）
            chinese-hubert-base/
            chinese-roberta-wwm-ext-large/
            gsv-v2final-pretrained/
            v2Pro/
            sv/
            s1v3.ckpt ...
        GPT_SoVITS/pretrained_models  → ../../models/pretrained  (symlink)

    幂等性：
        - 若 models/pretrained/ 已存在且原目录已是符号链接，则直接返回
        - 若原目录已是指向 models/pretrained/ 的符号链接，则只确保目标存在

    兼容性：
        - WSL (Linux)：os.symlink 直接支持
        - Colab (Linux)：同上
        - Windows：需要管理员权限或开发者模式，WSL 内运行时不受影响

    Returns
    -------
    Path
        迁移后的实际目录路径 models/pretrained/
    """
    src = PROJECT_ROOT / "GPT_SoVITS" / "pretrained_models"
    dst = Path(MODELS_DIR) / "pretrained"

    # ── 情况1：已完成迁移（src 是符号链接且目标正确） ─────────────────
    if src.is_symlink():
        current_target = Path(os.readlink(src))
        # 将相对路径转为绝对路径再比较
        if not current_target.is_absolute():
            current_target = (src.parent / current_target).resolve()
        if current_target.resolve() == dst.resolve():
            logger.info("预训练模型目录已完成迁移，无需重复操作: %s", dst)
            return dst
        else:
            logger.warning(
                "符号链接已存在但指向不同目标: %s -> %s，跳过迁移",
                src,
                current_target,
            )
            return dst

    # ── 情况2：原目录不存在（Colab 首次运行，目录尚未下载） ──────────
    if not src.exists():
        logger.info("原目录不存在，直接创建 models/pretrained/ 并建立符号链接")
        dst.mkdir(parents=True, exist_ok=True)
        # 使用相对路径创建符号链接，提升可移植性
        rel_target = os.path.relpath(dst, src.parent)
        os.symlink(rel_target, src)
        logger.info("符号链接已创建: %s -> %s", src, rel_target)
        return dst

    # ── 情况3：原目录是真实目录，需要迁移 ────────────────────────────
    if src.is_dir():
        if dst.exists():
            logger.warning(
                "目标目录已存在: %s\n请手动确认后删除其中一个，再重新运行迁移。",
                dst,
            )
            return dst

        logger.info("开始迁移预训练模型目录...")
        logger.info("  源:  %s", src)
        logger.info("  目标: %s", dst)

        # 确保 models/ 目标父目录存在
        dst.parent.mkdir(parents=True, exist_ok=True)

        # 移动目录（跨文件系统时自动回退到 copy+delete）
        shutil.move(str(src), str(dst))
        logger.info("目录移动完成")

        # 在原位置创建相对符号链接
        rel_target = os.path.relpath(dst, src.parent)
        os.symlink(rel_target, src)
        logger.info("符号链接已创建: %s -> %s", src, rel_target)

        # 验证链接可访问
        if src.exists() and src.resolve() == dst.resolve():
            logger.info("迁移验证通过 ✓")
        else:
            logger.error("迁移后验证失败，请手动检查: %s", src)

        return dst

    raise RuntimeError(f"无法识别的路径状态: {src}")


def download_asr_models():
    logger.info("Downloading Faster Whisper ASR models...")
    # For automatic ASR labeling, Large V3 is recommended for top accuracy [7, 8]
    snapshot_download(
        repo_id="Systran/faster-whisper-large-v3",
        local_dir=os.path.join(MODELS_DIR, "asr", "faster-whisper-large-v3"),
    )


def run_download(use_proxy: bool = False):
    """
    Interactive download task with menu-based selection.

    Parameters
    ----------
    use_proxy : bool
        Whether to enable network proxy.
    """
    # 1. Setup logging
    setup_logging()

    # 2. Setup proxy (if requested)
    if use_proxy:
        setup_proxy()
    else:
        logger.info("Proxy disabled, connecting directly to download servers.")

    # 3. Ensure models root directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 4. Interactive download loop
    menu_template = """
========================================
    GPT-SoVITS Model Downloader Menu
========================================
Current Status: {proxy_status}
----------------------------------------
Enter command to download (case-insensitive):
- ALL      : Download all models (except Roformer)
- CORE     : GPT-SoVITS core pretrained models
- UVR5     : UVR5 traditional VR weights
- ROFORMER : High-quality Roformer models (extra)
- ASR      : Faster Whisper ASR models
- G2PW     : Chinese polyphone G2PW models
- MIGRATE  : Migrate pretrained_models/ to models/pretrained/ (symlink)
- PROXY    : Toggle network proxy ON/OFF
- EXIT     : Exit the downloader
========================================
"""

    while True:
        status_str = "Proxy ON" if use_proxy else "Proxy OFF"
        print(menu_template.format(proxy_status=status_str))

        cmd = (
            input("\nEnter command [CORE/UVR5/ROFORMER/ASR/G2PW/ALL/PROXY/EXIT]: ")
            .strip()
            .upper()
        )

        if cmd == "EXIT":
            logger.info("Exiting model downloader.")
            break

        if cmd == "PROXY":
            use_proxy = not use_proxy
            if use_proxy:
                setup_proxy()
                logger.info("Proxy settings applied.")
            else:
                for env_var in [
                    "http_proxy",
                    "https_proxy",
                    "HTTP_PROXY",
                    "HTTPS_PROXY",
                ]:
                    os.environ.pop(env_var, None)
                logger.info("Proxy settings cleared.")
            continue

        try:
            if cmd == "CORE":
                download_gpt_sovits_models()
            elif cmd == "UVR5":
                download_uvr5_weights()
            elif cmd == "ROFORMER":
                download_roformer_weights()
            elif cmd == "ASR":
                download_asr_models()
            elif cmd == "G2PW":
                download_g2pw_models()
            elif cmd == "MIGRATE":
                migrate_pretrained_dir()
            elif cmd == "ALL":
                logger.info("Starting bulk download of pretrained models...")
                download_gpt_sovits_models()
                download_uvr5_weights()
                download_asr_models()
                download_g2pw_models()
            else:
                print(f"Unknown command: {cmd}, please try again.")
                continue

            logger.info("Task [%s] completed successfully!", cmd)
        except Exception as e:
            logger.error("Error during task [%s]: %s", cmd, e)
            logger.error(
                "Please check network or proxy settings (use_proxy=%s).", use_proxy
            )


if __name__ == "__main__":
    # Default is proxy OFF, suitable for Colab.
    # For local usage needing proxy, call run_download(use_proxy=True) instead.
    run_download(use_proxy=False)
