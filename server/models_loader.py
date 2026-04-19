import logging
import os
import zipfile

from huggingface_hub import hf_hub_download, snapshot_download

from server.logger import setup_logging
from server.proxy import setup_proxy

logger = logging.getLogger("server.models_loader")


def download_gpt_sovits_models():
    logger.info("正在下载 GPT-SoVITS 核心预训练模型...")
    # 核心预训练模型：包含 GPT 和 SoVITS 的基础权重 [1, 2]
    snapshot_download(
        repo_id="lj1995/GPT-SoVITS",
        local_dir="GPT_SoVITS/pretrained_models",
        allow_patterns=[
            "chinese-hubert-base/*",
            "chinese-roberta-wwm-ext-large/*",
            "s1v2base_23k.ckpt",
            "s2G488k.pth",
            "gsv-v2final-pretrained/*",  # 如果需要 V2 模型
        ],
    )


def download_g2pw_models():
    logger.info("正在下载 G2PW 中文多音字对齐模型...")
    # 用于提升中文 TTS 的多音字消歧准确性 [3, 4]
    # 注意：如果 HF 仓库路径不同，请根据实际调整 repo_id
    snapshot_download(repo_id="lj1995/G2PWModel", local_dir="GPT_SoVITS/text/G2PWModel")


def download_uvr5_weights():
    logger.info("正在下载 UVR5 传统 VR 模型权重...")
    # 用于从原始音频中提取纯净干声 [5, 6]
    uvr5_models = [
        "HP2_all_vocals.pth",
        "HP5_only_main_vocal.pth",
        "VR-DeEchoAggressive.pth",
        "VR-DeEchoNormal.pth",
        "Onnx_Aggressive_DeReverb.pth",
    ]
    for model in uvr5_models:
        hf_hub_download(
            repo_id="lj1995/UVR5", filename=model, local_dir="models/uvr5"
        )


def download_roformer_weights():
    """
    下载 BS-Roformer 和 Mel-Band-Roformer 模型权重及配套配置文件。

    从 XXXXRT/GPT-SoVITS-Pretrained 下载 uvr5_weights.zip，
    再从压缩包中提取 Roformer 相关文件到 models/uvr5/。
    每个模型由 .ckpt（权重）和 .yaml（配置）两个文件组成，
    文件名（除后缀外）必须完全一致，且文件名中须包含 'roformer'。
    """
    logger.info("正在下载 uvr5_weights.zip (XXXXRT/GPT-SoVITS-Pretrained)...")
    zip_path = hf_hub_download(
        repo_id="XXXXRT/GPT-SoVITS-Pretrained",
        filename="uvr5_weights.zip",
        cache_dir="models/_cache",
    )

    # 需要从压缩包中提取的 Roformer 文件（.ckpt 与 .yaml 必须成对）
    roformer_targets = {
        "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        "model_bs_roformer_ep_368_sdr_12.9628.yaml",
        "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
        "model_bs_roformer_ep_317_sdr_12.9755.yaml",
        "kim_mel_band_roformer.ckpt",
        "kim_mel_band_roformer.yaml",
    }

    os.makedirs("models/uvr5", exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for entry in zf.namelist():
            basename = os.path.basename(entry)
            if basename in roformer_targets:
                out_path = os.path.join("models/uvr5", basename)
                with zf.open(entry) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                extracted.append(basename)
                logger.info("已提取: %s", basename)

    if not extracted:
        logger.warning("压缩包中未找到 Roformer 模型文件，请手动放置到 models/uvr5/")
    else:
        logger.info("Roformer 模型提取完成，共 %d 个文件 -> models/uvr5/", len(extracted))


def download_asr_models():
    logger.info("正在下载 Faster Whisper ASR 模型...")
    # 用于自动语音识别打标，推荐 Large V3 以获得最高准确率 [7, 8]
    snapshot_download(
        repo_id="Systran/faster-whisper-large-v3",
        local_dir="tools/asr/models/faster-whisper-large-v3",
    )


if __name__ == "__main__":
    # 1. 开启代理
    setup_logging()
    setup_proxy()

    # 2. 创建必要目录（路径与 download_uvr5_weights 保持一致）
    # os.makedirs("GPT_SoVITS/pretrained_models", exist_ok=True)
    # os.makedirs("GPT_SoVITS/text/G2PWModel", exist_ok=True)
    os.makedirs("models/uvr5", exist_ok=True)
    # os.makedirs("tools/asr/models", exist_ok=True)

    # 3. 执行下载
    try:
        # download_gpt_sovits_models()
        # download_g2pw_models()
        download_uvr5_weights()
        # download_asr_models()
        logger.info("所有模型下载并配置完成！你可以开始集成 FastAPI 服务了。")
    except Exception as e:
        logger.error("下载过程中出现错误: %s", e)
        logger.error("请检查网络连接或代理设置。")
