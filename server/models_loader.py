import logging
import os
import zipfile

from huggingface_hub import hf_hub_download, snapshot_download

from server.config import MODELS_DIR
from server.logger import setup_logging
from server.proxy import setup_proxy

logger = logging.getLogger("server.models_loader")


def download_gpt_sovits_models():
    logger.info("Downloading GPT-SoVITS core pretrained models...")
    # Core pretrained models: foundation weights for GPT and SoVITS [1, 2]
    snapshot_download(
        repo_id="lj1995/GPT-SoVITS",
        local_dir=os.path.join(MODELS_DIR, "pretrained_models"),
        allow_patterns=[
            "chinese-hubert-base/*",
            "chinese-roberta-wwm-ext-large/*",
            "s1v2base_23k.ckpt",
            "s2G488k.pth",
            "gsv-v2final-pretrained/*",  # If V2 models are needed
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
