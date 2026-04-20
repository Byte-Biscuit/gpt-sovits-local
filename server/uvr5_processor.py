"""
UVR5 Vocal Separation Module

Responsibilities:
    Performs vocal/instrumental separation on sliced audio to output clean vocals for training.

Directory Conventions:
    models/uvr5/                             ← Model weights storage directory
    assets/{speaker}/sliced/          ← Input (sliced results)
    assets/{speaker}/vocals/          ← Output vocals (sent to ASR for labeling)
    assets/{speaker}/instrumental/    ← Output accompaniment (can be discarded during training)

Recommended Processing Chain (Single Character Cloning):
    1. HP5_only_main_vocal  → Separate lead vocals
    2. VR-DeEchoNormal      → De-reverb (can be skipped if the recording environment is clean)

Model Types and Corresponding Classes:
    HP2_all_vocals / HP5_only_main_vocal       → AudioPre (vr.py)
    VR-DeEchoNormal / VR-DeEchoAggressive     → AudioPreDeEcho (vr.py)
    bs_roformer_* / *mel_band_roformer*        → Roformer_Loader (bsroformer.py)
"""

import logging
import os

import torch

from server.config import MODELS_DIR
from server.logger import setup_logging  # noqa: E402
from tools.uvr5.bsroformer import Roformer_Loader  # noqa: E402
from tools.uvr5.vr import AudioPre, AudioPreDeEcho  # noqa: E402

logger = logging.getLogger("server.uvr5_processor")

# Model weights are uniformly stored in the uvr5 directory under config.MODELS_DIR
UVR5_MODELS_DIR = os.path.join(MODELS_DIR, "uvr5")

# Model filenames (corresponding to filenames in the MODELS_DIR/uvr5/ directory)
MODEL_HP5 = "HP5_only_main_vocal.pth"  # Lead vocals only, preferred for training data
MODEL_HP2 = "HP2_all_vocals.pth"  # All vocals (including backing vocals)
MODEL_DEECHO_NORMAL = "VR-DeEchoNormal.pth"  # Gentle de-reverb
MODEL_DEECHO_AGGRESSIVE = "VR-DeEchoAggressive.pth"  # Aggressive de-reverb
# Roformer series (high quality, requires GPU, .ckpt and same-named .yaml must be stored in pairs)
MODEL_BS_ROFORMER = (
    "model_bs_roformer_ep_317_sdr_12.9755.ckpt"  # Highest overall quality
)
MODEL_MEL_ROFORMER = (
    "kim_mel_band_roformer.ckpt"  # Best vocal detail, preferred for TTS
)


def _detect_device() -> tuple[str, bool]:
    """
    Automatically select the inference device.

    Returns
    -------
    device : str
        "cuda" or "cpu"
    is_half : bool
        Enable half precision (FP16) on GPU to save VRAM; forced False on CPU
    """
    if torch.cuda.is_available():
        return "cuda", True
    return "cpu", False


class UVR5Processor:
    """
    Encapsulates UVR5 vocal separation workflow.

    Models are lazy-loaded (only loaded into VRAM on first call) to avoid unnecessary VRAM usage.

    Parameters
    ----------
    models_dir : str
        Model weights directory, default models/uvr5/
    agg : int
        Vocal extraction aggressiveness (0~20), higher is more aggressive but may degrade quality, default 10
    output_format : str
        Output audio format, "wav" or "flac", default "wav"
    device : str | None
        Inference device, None for auto-detection
    is_half : bool | None
        Whether to use half precision, None for auto (True on GPU, False on CPU)
    """

    def __init__(
        self,
        models_dir: str = UVR5_MODELS_DIR,
        agg: int = 10,
        output_format: str = "wav",
        device: str | None = None,
        is_half: bool | None = None,
    ):
        self.models_dir = os.path.abspath(models_dir)
        self.agg = agg
        self.output_format = output_format

        # Auto-detect or use specified device
        auto_device, auto_half = _detect_device()
        self.device = device if device is not None else auto_device
        self.is_half = is_half if is_half is not None else auto_half

        logger.info(
            "UVR5Processor Init | Device=%s | HalfPrecision=%s | ModelsDir=%s",
            self.device,
            self.is_half,
            self.models_dir,
        )

        # Lazy load cache: avoid reloading the same model
        self._model_cache: dict[str, AudioPre | AudioPreDeEcho | Roformer_Loader] = {}

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _model_path(self, filename: str) -> str:
        """Joins model filename with models_dir and checks if file exists."""
        path = os.path.join(self.models_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                f"Please run 'python -m server.models_loader' to download model weights first."
            )
        return path

    def _load_model(
        self, model_filename: str
    ) -> AudioPre | AudioPreDeEcho | Roformer_Loader:
        """
        Lazy-loads model: reuses from cache if already loaded to avoid redundant memory usage.

        Model type inference priority (based on filename):
          1. Contains 'roformer'              → Roformer_Loader (bs_roformer / mel_band_roformer)
          2. Contains 'DeEcho' or 'DeReverb' → AudioPreDeEcho (de-reverb/de-echo)
          3. Others                           → AudioPre (standard vocal/instrumental separation)
        """
        if model_filename in self._model_cache:
            return self._model_cache[model_filename]

        path = self._model_path(model_filename)
        logger.info("Loading model: %s", model_filename)

        name_lower = model_filename.lower()
        if "roformer" in name_lower:
            # Roformer series requires same-named .yaml config file and .ckpt file stored in pairs
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
        logger.info("Model loaded successfully: %s", model_filename)
        return model

    def _run_separation(
        self,
        model_filename: str,
        input_dir: str,
        vocal_dir: str,
        ins_dir: str,
    ) -> list[str]:
        """
        Performs separation on all WAV files under input_dir, returns list of successfully processed files.

        Pass None to ins_dir to skip instrumental output and save disk space.
        """
        os.makedirs(vocal_dir, exist_ok=True)
        os.makedirs(ins_dir, exist_ok=True)

        model = self._load_model(model_filename)

        wav_files = sorted(
            f for f in os.listdir(input_dir) if f.lower().endswith(".wav")
        )
        if not wav_files:
            logger.warning("No WAV files found in input directory: %s", input_dir)
            return []

        processed = []
        for wav in wav_files:
            inp = os.path.join(input_dir, wav)
            logger.info("Processing: %s", inp)
            try:
                # Parameters and order of _path_audio_ differ among the three model classes:
                #   AudioPre:      (music_file, ins_root,   vocal_root, format)
                #   AudioPreDeEcho:(music_file, vocal_root, ins_root,   format)  ← vocal/ins order swapped
                #   Roformer_Loader:(input,    others_root, vocal_root, format)  ← parameter names differ
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
                logger.error("Separation failed: %s | Reason: %s", inp, exc)

        logger.info(
            "Separation complete, %d files processed -> %s", len(processed), vocal_dir
        )
        return processed

    # ------------------------------------------------------------------
    # Public API
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
        Performs vocal/instrumental separation on a speaker directory.

        Parameters
        ----------
        speaker_dir : str
            Speaker root directory, e.g., assets/wangliqun
        input_subdir : str
            Input subdirectory (sliced results), default "sliced"
        vocal_subdir : str
            Vocal output subdirectory, default "vocals"
        ins_subdir : str
            Instrumental output subdirectory, default "instrumental"
        model_filename : str
            Model filename to use, default model_bs_roformer_ep_317_sdr_12.9755.ckpt

        Returns
        -------
        list[str]
            List of successfully processed input file paths
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
        Directly specify input/output directories for vocal separation, without speaker_dir convention.

        Parameters
        ----------
        input_dir : str
            Directory of sliced WAV files (full path)
        vocal_dir : str
            Vocal output directory (full path, created if not exists)
        ins_dir : str
            Instrumental output directory (full path, created if not exists)
        model_filename : str
            Model filename to use, default MODEL_BS_ROFORMER

        Returns
        -------
        list[str]
            List of successfully processed input file paths
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
        Performs de-reverb/de-echo processing on the vocals directory (optional step).

        Parameters
        ----------
        speaker_dir : str
            Speaker root directory
        input_subdir : str
            Input subdirectory (vocal separation results), default "vocals"
        vocal_subdir : str
            De-reverbed output subdirectory, default "vocals_clean"
        ins_subdir : str
            Reverb component output subdirectory, default "vocals_echo"
        aggressive : bool
            True for aggressive de-reverb, False for gentle mode, default False
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
        """Releases VRAM for all loaded models, call after processing is finished."""
        self._model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model cache released")


if __name__ == "__main__":
    setup_logging()

    from server.config import ASSETS_DIR

    # ---- Interactive Speaker Name Input ----
    default_speaker = "wangliqun"
    user_input = input(
        f"Please enter speaker name (Press Enter to use default '{default_speaker}'): "
    ).strip()
    SPEAKER_NAME = user_input if user_input else default_speaker

    # Join using config.ASSETS_DIR
    SPEAKER_DIR = os.path.join(ASSETS_DIR, SPEAKER_NAME)

    # ---- Processing Parameters ----
    AGG = 10  # Vocal extraction aggressiveness 0~20
    REMOVE_ECHO = (
        False  # Whether to perform de-reverb (set False if recording is clean)
    )
    ECHO_MODE = False  # True=Aggressive de-reverb, False=Gentle de-reverb
    # ----------------------------------------

    processor = UVR5Processor(agg=AGG)

    # Step 1: Vocal/Instrumental Separation (BS-Roformer by default)
    logger.info("Starting audio separation: %s", SPEAKER_DIR)
    processor.separate_vocals(SPEAKER_DIR)

    # Step 2: De-reverb (Optional)
    if REMOVE_ECHO:
        processor.remove_echo(SPEAKER_DIR, aggressive=ECHO_MODE)

    # Release VRAM
    processor.release_models()
    logger.info("All processing completed!")
