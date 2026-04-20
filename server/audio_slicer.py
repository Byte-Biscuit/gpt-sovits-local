"""
Audio slicing utility for GPT-SoVITS training data preparation.

Slices long WAV files into short segments (typically 3~15 s) based on silence
detection, suitable for use as training data.

Directory convention:
    assets/{speaker}/          <- source WAV files
    assets/{speaker}/sliced/   <- output segments

Reuses tools/slicer2.Slicer from the original GPT-SoVITS codebase.
"""

import logging
import os
import traceback

import ffmpeg
import numpy as np
from scipy.io import wavfile

from server.config import ASSETS_DIR
from server.logger import setup_logging
from tools.slicer2 import Slicer  # noqa: E402

# 固定使用完整模块路径作为 logger 名称，避免直接运行时 __name__ == '__main__' 导致
# 日志脱离 'server' 根节点，无法被 setup_logging() 配置的 handler 接管
logger = logging.getLogger("server.audio_slicer")

SAMPLE_RATE = 32000  # Hz – required by Slicer / GPT-SoVITS pipeline


def _load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file to a mono float32 numpy array via ffmpeg."""
    path = path.strip().strip("'\"")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        out, _ = (
            ffmpeg.input(path, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as exc:
        raise RuntimeError(f"Failed to load audio: {path}") from exc
    return np.frombuffer(out, np.float32).flatten()


class AudioSlicer:
    """
    Slice WAV files under a speaker directory into short training segments.

    Parameters
    ----------
    threshold : float
        Volume (dB) below which a frame is considered silence. Default -34.
    min_length : int
        Minimum segment length in milliseconds. Default 4000 (4 s).
    min_interval : int
        Minimum silence interval (ms) used as a cut point. Default 300.
    hop_size : int
        Frame hop size (ms) for RMS computation. Default 10.
    max_sil_kept : int
        Maximum silence retained at segment edges (ms). Default 500.
    max_amp : float
        Peak amplitude after normalisation. Default 0.9.
    alpha : float
        Blend ratio between normalised and original audio. Default 0.25.
    """

    def __init__(
        self,
        threshold: float = -34.0,
        min_length: int = 4000,
        min_interval: int = 300,
        hop_size: int = 10,
        max_sil_kept: int = 500,
        max_amp: float = 0.9,
        alpha: float = 0.25,
    ):
        self.threshold = threshold
        self.min_length = min_length
        self.min_interval = min_interval
        self.hop_size = hop_size
        self.max_sil_kept = max_sil_kept
        self.max_amp = max_amp
        self.alpha = alpha

        self._slicer = Slicer(
            sr=SAMPLE_RATE,
            threshold=threshold,
            min_length=min_length,
            min_interval=min_interval,
            hop_size=hop_size,
            max_sil_kept=max_sil_kept,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def slice_file(self, input_path: str, output_dir: str) -> list[str]:
        """
        Slice a single WAV file and write segments to *output_dir*.

        Returns a list of written output file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        written: list[str] = []
        name = os.path.basename(input_path)
        try:
            audio = _load_audio(input_path, SAMPLE_RATE)
            for idx, (chunk, start, end) in enumerate(self._slicer.slice(audio)):
                tmp_max = np.abs(chunk).max()
                if tmp_max > 1:
                    chunk = chunk / tmp_max
                chunk = (chunk / tmp_max * (self.max_amp * self.alpha)) + (
                    1 - self.alpha
                ) * chunk
                out_path = os.path.join(
                    output_dir, f"{name}_{idx:04d}_{start:010d}_{end:010d}.wav"
                )
                wavfile.write(out_path, SAMPLE_RATE, (chunk * 32767).astype(np.int16))
                written.append(out_path)
        except Exception:
            logger.error("切片失败: %s\n%s", input_path, traceback.format_exc())
        return written

    def slice_speaker(
        self, speaker_dir: str, output_subdir: str = "sliced"
    ) -> list[str]:
        """
        Slice all WAV files inside *speaker_dir*.

        Segments are written to ``{speaker_dir}/{output_subdir}/``.

        Returns a list of all written output file paths.
        """
        speaker_dir = os.path.abspath(speaker_dir)
        output_dir = os.path.join(speaker_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        wav_files = sorted(
            f for f in os.listdir(speaker_dir) if f.lower().endswith(".wav")
        )
        if not wav_files:
            logger.warning("目录下未找到 WAV 文件: %s", speaker_dir)
            return []

        all_written: list[str] = []
        for wav in wav_files:
            inp = os.path.join(speaker_dir, wav)
            logger.info("正在切片: %s", inp)
            written = self.slice_file(inp, output_dir)
            logger.info("  -> 生成 %d 个片段", len(written))
            all_written.extend(written)

        logger.info("切片完成，共 %d 个片段  ->  %s", len(all_written), output_dir)
        return all_written


if __name__ == "__main__":
    # Use unified logging config, output to file + console
    setup_logging()

    # ---- Interactive Speaker Name Input ----
    default_speaker = "wangliqun"
    user_input = input(
        f"Please enter speaker name (Press Enter to use default '{default_speaker}'): "
    ).strip()
    SPEAKER_NAME = user_input if user_input else default_speaker

    # Path joining with config.ASSETS_DIR
    SPEAKER_DIR = os.path.join(ASSETS_DIR, SPEAKER_NAME)

    # ---- Other Processing Parameters ----
    THRESHOLD = -34.0  # Silence threshold (dB)
    MIN_LENGTH = 4000  # Minimum segment length (ms)
    MIN_INTERVAL = 300  # Minimum cut interval (ms)
    HOP_SIZE = 10  # RMS hop size (ms)
    MAX_SIL_KEPT = 500  # Silence kept at edges (ms)
    MAX_AMP = 0.9  # Normalization peak
    ALPHA = 0.25  # Normalization blend ratio
    # ----------------------------------------

    logger.info("Starting speaker processing: %s", SPEAKER_DIR)
    slicer = AudioSlicer(
        threshold=THRESHOLD,
        min_length=MIN_LENGTH,
        min_interval=MIN_INTERVAL,
        hop_size=HOP_SIZE,
        max_sil_kept=MAX_SIL_KEPT,
        max_amp=MAX_AMP,
        alpha=ALPHA,
    )
    results = slicer.slice_speaker(SPEAKER_DIR)
    logger.info("All finished, generated %d segments", len(results))
