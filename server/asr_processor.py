"""
ASR (Automatic Speech Recognition) Labeling Module

Responsibilities:
    Performs ASR transcription on clean vocals separated by UVR5 to generate
    annotation list files required for GPT-SoVITS training.

Directory Conventions:
    assets/{speaker}/vocals/         ← Input (vocals after UVR5 separation)
    assets/{speaker}/asr/            ← Output directory
    assets/{speaker}/asr/{speaker}.list  ← Annotation file (Format: audio_path|speaker|lang|text)

Supported Backends:
    Faster Whisper  → Multilingual, supports external model paths (ideal for Google Drive / Colab)
    FunASR (Damo)   → Specialized for Chinese, requires three sub-models from ModelScope

Annotation Format (GPT-SoVITS Standard):
    /abs/path/to/segment.wav|speaker_name|ZH|transcribed_text
"""

import logging
import os
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

ASR_TOOLS_PATH = str(Path(__file__).parent.parent / "tools" / "asr")

if ASR_TOOLS_PATH not in sys.path:
    sys.path.append(ASR_TOOLS_PATH)

import torch  # noqa: E402

from server.config import ASSETS_DIR, MODELS_DIR  # noqa: E402
from server.logger import setup_logging  # noqa: E402

logger = logging.getLogger("server.asr_processor")

DEFAULT_WHISPER_MODEL_DIR = os.path.join(MODELS_DIR, "asr", "faster-whisper-large-v3")


def _detect_device() -> tuple[str, str]:
    """
    Automatically selects inference device and corresponding precision.

    Returns
    -------
    device : str
        "cuda" or "cpu"
    precision : str
        float16 for GPU to save VRAM; int8 for CPU acceleration
    """
    if torch.cuda.is_available():
        return "cuda", "float16"
    return "cpu", "int8"


class ASRProcessor:
    """
    Encapsulates ASR labeling flow, supporting Faster Whisper and FunASR backends.

    Parameters
    ----------
    model_path : str
        Faster Whisper model directory.
        Local: Defaults to MODELS_DIR/asr/faster-whisper-large-v3
        Colab: Use Google Drive mount path, e.g.,
        /content/drive/MyDrive/gpt-sovits/models/asr/faster-whisper-large-v3
    language : str
        Target language code. "zh", "en", "ja", or "auto" for detection.
        For Chinese, Whisper switching to FunASR is recommended for higher accuracy.
    precision : str | None
        Inference precision. None for auto (GPU→float16, CPU→int8).
    device : str | None
        Inference device. None for auto-detection.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_WHISPER_MODEL_DIR,
        language: str = "zh",
        precision: str | None = None,
        device: str | None = None,
    ):
        self.model_path = os.path.abspath(model_path)
        self.language = language

        auto_device, auto_precision = _detect_device()
        self.device = device if device is not None else auto_device
        self.precision = precision if precision is not None else auto_precision

        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"Whisper model directory not found: {self.model_path}\n"
                "Please download the model locally first or provide the Google Drive path.\n"
                "Example (Colab): model_path='/content/drive/MyDrive/gpt-sovits/models/asr/faster-whisper-large-v3'"
            )

        logger.info(
            "ASRProcessor Init | Device=%s | Precision=%s | Language=%s | Model=%s",
            self.device,
            self.precision,
            self.language,
            self.model_path,
        )

        # Lazy loading to avoid CUDA initialization on import
        self._model = None

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _load_model(self):
        """Lazy loads Whisper model, reusing the instance for subsequent calls."""
        if self._model is not None:
            return self._model

        from faster_whisper import WhisperModel

        logger.info("Loading Whisper model: %s", self.model_path)
        self._model = WhisperModel(
            self.model_path, device=self.device, compute_type=self.precision
        )
        logger.info("Whisper model loaded successfully")
        return self._model

    def _transcribe_file(self, file_path: str, speaker: str) -> str | None:
        """
        Transcribes a single audio file and returns the annotation line.

        Format: {absolute_path}|{speaker}|{language_upper}|{text}
        Returns None on failure.
        """
        model = self._load_model()
        language = self.language if self.language != "auto" else None

        try:
            segments, info = model.transcribe(
                audio=file_path,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=700),
                language=language,
            )

            text = ""
            detected_lang = info.language.lower() if info.language else ""

            # If Chinese/Cantonese is detected, switch to FunASR for better results
            if detected_lang in ("zh", "yue"):
                logger.debug(
                    "Chinese detected, switching to FunASR: %s",
                    os.path.basename(file_path),
                )
                try:
                    from tools.asr.funasr_asr import only_asr

                    text = only_asr(file_path, language=detected_lang)
                except Exception:
                    logger.warning(
                        "FunASR failed, falling back to Whisper: %s",
                        os.path.basename(file_path),
                    )

            # If FunASR is unavailable or doesn't return text, use Whisper output
            if not text:
                for segment in segments:
                    text += segment.text

            text = text.strip()
            if not text:
                logger.warning(
                    "Transcription result is empty, skipping: %s",
                    os.path.basename(file_path),
                )
                return None

            lang_label = (detected_lang or self.language).upper()
            abs_path = os.path.abspath(file_path)
            return f"{abs_path}|{speaker}|{lang_label}|{text}"

        except Exception:
            logger.error(
                "Transcription failed: %s\n%s", file_path, traceback.format_exc()
            )
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe_speaker(
        self,
        speaker_dir: str,
        input_subdir: str = "vocals",
        output_subdir: str = "asr",
    ) -> str | None:
        """
        Performs ASR labeling for all vocal files in a speaker directory.

        Parameters
        ----------
        speaker_dir : str
            Root directory for the speaker, e.g., assets/wangliqun
        input_subdir : str
            Subdirectory containing clean vocals, default 'vocals'
        output_subdir : str
            Subdirectory for annotation output, default 'asr'

        Returns
        -------
        str | None
            Absolute path to the generated .list file; None if no results
        """
        speaker_dir = os.path.abspath(speaker_dir)
        speaker = os.path.basename(speaker_dir)
        input_dir = os.path.join(speaker_dir, input_subdir)
        output_dir = os.path.join(speaker_dir, output_subdir)

        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Vocals directory not found: {input_dir}")

        wav_files = sorted(
            f for f in os.listdir(input_dir) if f.lower().endswith(".wav")
        )
        if not wav_files:
            logger.warning("No WAV files found in vocals directory: %s", input_dir)
            return None

        logger.info(
            "Starting ASR Labeling | Speaker=%s | Count=%d", speaker, len(wav_files)
        )

        lines = []
        failed = 0
        with tqdm(total=len(wav_files), desc=f"ASR [{speaker}]", unit="file") as pbar:
            for wav in wav_files:
                file_path = os.path.join(input_dir, wav)
                result = self._transcribe_file(file_path, speaker)
                if result:
                    lines.append(result)
                else:
                    failed += 1
                pbar.set_postfix(success=len(lines), failed=failed)
                pbar.update(1)

        if not lines:
            logger.warning("All files failed or returned empty results")
            return None

        os.makedirs(output_dir, exist_ok=True)
        list_path = os.path.join(output_dir, f"{speaker}.list")
        with open(list_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(
            "ASR Labeling Complete | Success=%d/%d | List -> %s",
            len(lines),
            len(wav_files),
            list_path,
        )
        return list_path

    def transcribe_from_dir(
        self,
        input_dir: str,
        output_dir: str,
        speaker: str,
    ) -> str | None:
        """
        Performs ASR labeling with custom input/output directories.

        Parameters
        ----------
        input_dir : str
            Full path to input WAV files
        output_dir : str
            Full path to output directory (created if missing)
        speaker : str
            Speaker name (used in column 2 and as filename)

        Returns
        -------
        str | None
            Absolute path to the generated .list file
        """
        input_dir = os.path.abspath(input_dir)
        output_dir = os.path.abspath(output_dir)

        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        wav_files = sorted(
            f for f in os.listdir(input_dir) if f.lower().endswith(".wav")
        )
        if not wav_files:
            logger.warning("No WAV files found in directory: %s", input_dir)
            return None

        logger.info(
            "Starting ASR Labeling | Speaker=%s | Count=%d", speaker, len(wav_files)
        )

        lines = []
        failed = 0
        with tqdm(total=len(wav_files), desc=f"ASR [{speaker}]", unit="file") as pbar:
            for wav in wav_files:
                file_path = os.path.join(input_dir, wav)
                result = self._transcribe_file(file_path, speaker)
                if result:
                    lines.append(result)
                else:
                    failed += 1
                pbar.set_postfix(success=len(lines), failed=failed)
                pbar.update(1)

        if not lines:
            logger.warning("All files failed or returned empty results")
            return None

        os.makedirs(output_dir, exist_ok=True)
        list_path = os.path.join(output_dir, f"{speaker}.list")
        with open(list_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(
            "ASR Labeling Complete | Success=%d/%d | List -> %s",
            len(lines),
            len(wav_files),
            list_path,
        )
        return list_path

    def release_model(self) -> None:
        """Releases model from memory to save VRAM."""
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("ASR model cache released")


if __name__ == "__main__":
    setup_logging()

    # ---- Interactive Inputs ----
    default_speaker = "wangliqun"
    user_speaker = input(
        f"Please enter speaker name (Press Enter to use default '{default_speaker}'): "
    ).strip()
    SPEAKER_NAME = user_speaker if user_speaker else default_speaker

    default_lang = "zh"
    user_lang = input(
        f"Please enter language (zh/en/ja/auto) [Default: {default_lang}]: "
    ).strip()
    LANGUAGE = user_lang if user_lang else default_lang

    # Use ASSETS_DIR from config
    SPEAKER_DIR = os.path.join(ASSETS_DIR, SPEAKER_NAME)

    # Initialize processor with default model path from config
    # To use a custom path (e.g., in Colab), pass it to the constructor
    processor = ASRProcessor(language=LANGUAGE)

    try:
        list_file = processor.transcribe_speaker(SPEAKER_DIR)
        if list_file:
            logger.info("标注文件已生成: %s", list_file)
    finally:
        processor.release_model()
