import logging
import os
import shutil

from server.config import MODELS_DIR, TMP_MODELS_DIR
from server.logger import setup_logging

logger = logging.getLogger("server.sync_models")


def sync_models():
    """Sync final usable models from TMP_MODELS_DIR to MODELS_DIR.
    Ignore intermediate `G_*.pth` and `D_*.pth` files.
    """
    tmp_speaker_dir = os.path.join(TMP_MODELS_DIR, "speaker")
    tgt_speaker_dir = os.path.join(MODELS_DIR, "speaker")

    if not os.path.exists(tmp_speaker_dir):
        logger.warning(
            f"Source directory {tmp_speaker_dir} does not exist. Nothing to sync."
        )
        return

    os.makedirs(tgt_speaker_dir, exist_ok=True)

    for speaker in os.listdir(tmp_speaker_dir):
        src_dir = os.path.join(tmp_speaker_dir, speaker)
        if not os.path.isdir(src_dir):
            continue

        tgt_dir = os.path.join(tgt_speaker_dir, speaker)
        os.makedirs(tgt_dir, exist_ok=True)

        # Only copy .ckpt (GPT) and .pth (SoVITS final output) directly in the speaker root folder.
        # Exclude subdirectories (intermediate outputs usually stored inside `logs_{speaker}` folder, etc. if any)
        # Even if they are in root, exclude G_*.pth and D_*.pth.
        count = 0
        for item in os.listdir(src_dir):
            if not os.path.isfile(os.path.join(src_dir, item)):
                continue

            # Match extensions and ignore G/D temp models
            if item.endswith(".ckpt") or item.endswith(".pth"):
                if item.startswith("G_") or item.startswith("D_"):
                    # skip intermediate models
                    continue

                src_file = os.path.join(src_dir, item)
                tgt_file = os.path.join(tgt_dir, item)

                logger.info(f"Syncing {src_file} -> {tgt_file}")
                shutil.copy2(src_file, tgt_file)
                count += 1

        if count == 0:
            logger.info(f"No models to sync for speaker {speaker}")
        else:
            logger.info(f"Successfully synced {count} models for speaker {speaker}")


if __name__ == "__main__":
    setup_logging()
    sync_models()
