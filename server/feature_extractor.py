"""
特征提取模块 —— GPT-SoVITS 训练数据预处理（Step 3/4）

【在整个训练流程中的位置】
    切片 → 人声分离 → ASR 标注 → 【特征提取（本模块）】 → SoVITS/GPT 微调

【为什么需要特征提取】
    原始 WAV + 文本文本不能直接送入神经网络，需要先转换成模型可理解的向量表示。
    本模块完成从"人类可读数据"到"模型可训练张量"的全部转换。

【四步特征提取原理】

Step 1 — BERT 文本特征 (1-get-text.py)
    原理：使用 chinese-roberta-wwm-ext-large 将文字序列转成上下文语义向量。
         文本先经过 G2P（文字→音素）转换，再用 BERT 为每个音素计算嵌入向量。
    输出：logs/{exp}/2-name2text.txt  —— 音素序列
          logs/{exp}/3-bert/{name}.pt  —— 每个音素对应的 BERT 向量 (shape: 1024, N)
    作用：GPT 训练时提供"这句话说什么"的语义上下文，使韵律更自然。

Step 2 — HuBERT 音频语义特征 (2-get-hubert-wav32k.py)
    原理：HuBERT（Hidden-Unit BERT）是一种自监督音频模型，通过掩码预测学习音频的
         离散化语义表示，类似于语音版 BERT。输入 16kHz 单声道音频，输出每帧的
         768 维特征向量（25帧/秒）。
    输出：logs/{exp}/4-cnhubert/{name}.pt  —— HuBERT 特征 (shape: 1, 768, T)
          logs/{exp}/5-wav32k/{name}.wav   —— 重采样到 32kHz 的音频（SoVITS 训练用）
    作用：连接 GPT 和 SoVITS 的"语义桥梁"，后续量化为语义 token。

Step 3 — 说话人验证特征 (2-get-sv.py)  [v2Pro/v2ProPlus 专有]
    原理：使用 ERes2NetV2（ECAPA-TDNN 增强版）提取说话人音色指纹向量（d-vector）。
         与 HuBERT 的"说了什么"不同，SV 特征描述"谁在说话"。
    输出：logs/{exp}/7-sv_cn/{name}.pt  —— 说话人嵌入向量
    作用：训练时约束 SoVITS 始终贴近目标音色，v2Pro 相比 v2 音色稳定性显著提升。

Step 4 — 语义 Token 量化 (3-get-semantic.py)
    原理：将连续的 HuBERT 特征通过 SoVITS 预训练模型的 VQ（矢量量化）层，
         离散化为整数 token 序列（词汇表大小约 1024）。
         本质是把浮点音频特征"压缩"成类似文字 token 的离散表示。
    输出：logs/{exp}/6-name2semantic.tsv  —— 每行：文件名\\t语义token序列
    作用：GPT 的训练目标——给定文本，预测这个 token 序列。

【模型路径约定（v2Pro）】
    BERT:    GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/
    HuBERT:  GPT_SoVITS/pretrained_models/chinese-hubert-base/
    SV:      GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt
    SoVITS:  GPT_SoVITS/pretrained_models/v2Pro/s2Gv2Pro.pth
    配置:    GPT_SoVITS/configs/s2.json

【输出目录约定】
    logs/{speaker}/
        2-name2text.txt          ← 音素序列文本
        3-bert/{name}.pt         ← BERT 特征向量
        4-cnhubert/{name}.pt     ← HuBERT 特征向量
        5-wav32k/{name}.wav      ← 32kHz 重采样音频
        6-name2semantic.tsv      ← 量化语义 token
        7-sv_cn/{name}.pt        ← 说话人验证向量（v2Pro）
"""

import logging
import os
import shutil
import sys
import traceback
from pathlib import Path
from time import time as ttime

import numpy as np
import torch
from scipy.io import wavfile
from tqdm import tqdm

_project_root = str(Path(__file__).parent.parent.absolute())
_gpt_sovits_dir = os.path.join(_project_root, "GPT_SoVITS")
_eres2net_dir = os.path.join(_gpt_sovits_dir, "eres2net")

for _p in [_project_root, _gpt_sovits_dir, _eres2net_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from server.config import ASSETS_DIR, PRETRAINED_DIR, get_speaker_logs_dir  # noqa: E402
from server.logger import setup_logging  # noqa: E402

logger = logging.getLogger("server.feature_extractor")

# ── 预训练模型路径（从 config.PRETRAINED_DIR 派生） ──────────────────────────
BERT_DIR = os.path.join(PRETRAINED_DIR, "chinese-roberta-wwm-ext-large")
HUBERT_DIR = os.path.join(PRETRAINED_DIR, "chinese-hubert-base")
SV_MODEL_PATH = os.path.join(PRETRAINED_DIR, "sv", "pretrained_eres2netv2w24s4ep4.ckpt")
# SoVITS 模型结构配置文件
S2_CONFIG_PATH = os.path.join(_gpt_sovits_dir, "configs", "s2.json")

# 归一化参数（与原始脚本保持一致）
_MAXX = 0.95
_ALPHA = 0.5


def _detect_device() -> tuple[str, bool]:
    """自动选择推理设备，返回 (device_str, is_half)。"""
    if torch.cuda.is_available():
        return "cuda:0", True
    return "cpu", False


def _my_save(tensor, path: str) -> None:
    """
    绕过 torch.save 不支持中文路径的问题：先保存到临时文件再 move。
    与原始 prepare_datasets 脚本保持一致。
    """
    dir_ = os.path.dirname(path)
    tmp_path = os.path.join(dir_, f"_tmp_{ttime()}.pth")
    torch.save(tensor, tmp_path)
    shutil.move(tmp_path, path)


class FeatureExtractor:
    """
    封装 GPT-SoVITS 训练前的全部特征提取流程（4 步）。

    设计原则：
        - 模型懒加载：每步模型首次使用时才载入，处理完后可主动释放
        - 断点续跑：若目标 .pt/.wav 已存在则跳过，支持中断后重新运行
        - 路径全部来自 server/config.py，本地/Colab 统一切换

    Parameters
    ----------
    speaker : str
        说话人名称，对应 assets/{speaker}/ 目录
    version : str
        模型版本，当前推荐 "v2Pro"，影响 step3(SV) 和 step4(SoVITS 权重选择)
    device : str | None
        推理设备，None 自动检测
    is_half : bool | None
        是否半精度，None 自动（GPU→True, CPU→False）
    """

    def __init__(
        self,
        speaker: str,
        version: str = "v2Pro",
        device: str | None = None,
        is_half: bool | None = None,
    ):
        self.speaker = speaker
        self.version = version

        auto_device, auto_half = _detect_device()
        self.device = device if device is not None else auto_device
        self.is_half = is_half if is_half is not None else auto_half

        # 输入：ASR 标注文件
        self.list_path = os.path.join(ASSETS_DIR, speaker, "asr", f"{speaker}.list")
        # 输入：干声目录（步骤 2 重采样时从此读取）
        self.vocals_dir = os.path.join(ASSETS_DIR, speaker, "vocals")
        # 输出根目录
        self.exp_dir = get_speaker_logs_dir(speaker)

        # 各步骤输出子目录（与原始脚本命名完全一致，方便后续直接调用训练脚本）
        self.bert_dir = os.path.join(self.exp_dir, "3-bert")
        self.hubert_dir = os.path.join(self.exp_dir, "4-cnhubert")
        self.wav32_dir = os.path.join(self.exp_dir, "5-wav32k")
        self.sv_dir = os.path.join(self.exp_dir, "7-sv_cn")
        self.text_path = os.path.join(self.exp_dir, "2-name2text.txt")
        self.semantic_path = os.path.join(self.exp_dir, "6-name2semantic.tsv")

        for d in [
            self.exp_dir,
            self.bert_dir,
            self.hubert_dir,
            self.wav32_dir,
            self.sv_dir,
        ]:
            os.makedirs(d, exist_ok=True)

        logger.info(
            "FeatureExtractor 初始化 | 说话人=%s | 版本=%s | 设备=%s | 半精度=%s",
            speaker,
            version,
            self.device,
            self.is_half,
        )
        logger.info("实验输出目录: %s", self.exp_dir)

        # 懒加载缓存
        self._bert_model = None
        self._bert_tokenizer = None
        self._hubert_model = None
        self._sv_model = None
        self._vq_model = None

    # ──────────────────────────────────────────────────────────────────
    # Step 1：BERT 文本特征提取
    # ──────────────────────────────────────────────────────────────────

    def _load_bert(self):
        """懒加载 chinese-roberta-wwm-ext-large。"""
        if self._bert_model is not None:
            return self._bert_tokenizer, self._bert_model

        from transformers import AutoModelForMaskedLM, AutoTokenizer

        logger.info("加载 BERT 模型: %s", BERT_DIR)
        self._bert_tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
        self._bert_model = AutoModelForMaskedLM.from_pretrained(BERT_DIR)
        if self.is_half:
            self._bert_model = self._bert_model.half()
        self._bert_model = self._bert_model.to(self.device).eval()
        logger.info("BERT 模型加载完成")
        return self._bert_tokenizer, self._bert_model

    def extract_text_features(self) -> str:
        """
        Step 1：文本清洗 + BERT 音素级语义向量提取。

        读取 asr/{speaker}.list，将文本转为音素序列，
        并为中文文本提取 BERT 上下文向量。

        Returns
        -------
        str
            输出文本文件路径 (2-name2text.txt)
        """
        if not os.path.isfile(self.list_path):
            raise FileNotFoundError(f"ASR 标注文件不存在: {self.list_path}")

        from text.cleaner import clean_text

        from tools.my_utils import clean_path

        tokenizer, bert_model = self._load_bert()

        def _get_bert_feature(text: str, word2ph: list) -> torch.Tensor:
            """将文本转为音素级 BERT 特征（shape: 1024, N_phones）。"""
            if tokenizer is None:
                raise RuntimeError("BERT tokenizer is not initialized")
            with torch.no_grad():
                inputs = tokenizer(text, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                res = bert_model(**inputs, output_hidden_states=True)
                # 取最后第 3 层隐藏状态，去掉 [CLS] 和 [SEP]
                res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

            # 将词级特征扩展到音素级（每个字对应 word2ph[i] 个音素）
            phone_feats = []
            for i in range(len(word2ph)):
                phone_feats.append(res[i].repeat(word2ph[i], 1))
            return torch.cat(phone_feats, dim=0).T  # shape: (1024, N_phones)

        with open(self.list_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        results = []
        logger.info("Step 1/4 BERT 文本特征提取 | 共 %d 条", len(lines))

        with tqdm(total=len(lines), desc="Step1 BERT", unit="file") as pbar:
            for line in lines:
                try:
                    wav_path, spk, lang, text = line.split("|")
                    wav_name = os.path.basename(clean_path(wav_path))
                    # 语言代码统一转小写，与原始脚本一致
                    lang_lower = lang.lower()

                    phones, word2ph, norm_text = clean_text(
                        text.replace("%", "-").replace("￥", ","),
                        lang_lower,
                        version=self.version,
                    )

                    bert_path = os.path.join(self.bert_dir, f"{wav_name}.pt")
                    if not os.path.exists(bert_path) and lang_lower == "zh":
                        feat = _get_bert_feature(norm_text, word2ph)
                        assert feat.shape[-1] == len(phones), (
                            f"BERT 特征长度 {feat.shape[-1]} != 音素数 {len(phones)}"
                        )
                        _my_save(feat, bert_path)

                    results.append(
                        f"{wav_name}\t{' '.join(phones)}\t{word2ph}\t{norm_text}"
                    )
                except Exception:
                    logger.error(
                        "BERT 提取失败: %s\n%s", line[:80], traceback.format_exc()
                    )
                pbar.update(1)

        with open(self.text_path, "w", encoding="utf-8") as f:
            f.write("\n".join(results))

        logger.info("Step 1 完成 | 音素文件 -> %s", self.text_path)
        return self.text_path

    # ──────────────────────────────────────────────────────────────────
    # Step 2：HuBERT 音频语义特征提取
    # ──────────────────────────────────────────────────────────────────

    def _load_hubert(self):
        """懒加载 chinese-hubert-base。"""
        if self._hubert_model is not None:
            return self._hubert_model

        from feature_extractor import cnhubert

        cnhubert.cnhubert_base_path = HUBERT_DIR
        logger.info("加载 HuBERT 模型: %s", HUBERT_DIR)
        model = cnhubert.get_model()
        if self.is_half:
            model = model.half()
        self._hubert_model = model.to(self.device).eval()
        logger.info("HuBERT 模型加载完成")
        return self._hubert_model

    def extract_hubert_features(self) -> None:
        """
        Step 2：HuBERT 特征提取 + 32kHz 重采样。

        对每个干声文件：
          1. 重采样到 32kHz 并归一化 → 保存到 5-wav32k/
          2. 再降采样到 16kHz 送入 HuBERT → 保存 768 维特征到 4-cnhubert/
        """
        import librosa

        from tools.my_utils import load_audio

        if not os.path.isfile(self.list_path):
            raise FileNotFoundError(f"ASR 标注文件不存在: {self.list_path}")

        model = self._load_hubert()

        with open(self.list_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        logger.info("Step 2/4 HuBERT 特征提取 | 共 %d 条", len(lines))
        nan_fails = []

        with tqdm(total=len(lines), desc="Step2 HuBERT", unit="file") as pbar:
            for line in lines:
                try:
                    wav_path = line.split("|")[0]
                    wav_name = os.path.basename(wav_path)
                    hubert_path = os.path.join(self.hubert_dir, f"{wav_name}.pt")
                    wav32_path = os.path.join(self.wav32_dir, wav_name)

                    if os.path.exists(hubert_path) and os.path.exists(wav32_path):
                        pbar.update(1)
                        continue  # 断点续跑：已处理则跳过

                    # 加载音频到 32kHz float32 单声道
                    audio32 = load_audio(wav_path, 32000)
                    peak = np.abs(audio32).max()
                    if peak > 2.2:
                        logger.warning("幅度过大，跳过: %s (peak=%.2f)", wav_name, peak)
                        pbar.update(1)
                        continue

                    # 归一化（与原始脚本完全一致）
                    audio32_norm = (
                        audio32 / peak * (_MAXX * _ALPHA * 32768)
                        + (1 - _ALPHA) * 32768 * audio32
                    )
                    audio32_for_hubert = (
                        audio32 / peak * (_MAXX * _ALPHA * 1145.14)
                        + (1 - _ALPHA) * 1145.14 * audio32
                    )

                    # 降采样到 16kHz 供 HuBERT 使用
                    audio16 = librosa.resample(
                        audio32_for_hubert, orig_sr=32000, target_sr=16000
                    )
                    tensor16 = torch.from_numpy(audio16).unsqueeze(0)
                    if self.is_half:
                        tensor16 = tensor16.half()
                    tensor16 = tensor16.to(self.device)

                    ssl = (
                        model.model(tensor16)["last_hidden_state"].transpose(1, 2).cpu()
                    )

                    if np.isnan(ssl.detach().numpy()).sum() != 0:
                        nan_fails.append(wav_name)
                        logger.warning("NaN 特征，跳过: %s", wav_name)
                        pbar.update(1)
                        continue

                    # 保存 32kHz 音频（SoVITS 训练直接使用此文件）
                    wavfile.write(wav32_path, 32000, audio32_norm.astype("int16"))
                    # 保存 HuBERT 特征 (shape: 1, 768, T)
                    _my_save(ssl, hubert_path)

                except Exception:
                    logger.error(
                        "HuBERT 提取失败: %s\n%s", line[:80], traceback.format_exc()
                    )
                pbar.update(1)

        if nan_fails:
            logger.warning("NaN 过滤文件数: %d", len(nan_fails))
        logger.info(
            "Step 2 完成 | HuBERT -> %s | wav32k -> %s", self.hubert_dir, self.wav32_dir
        )

    # ──────────────────────────────────────────────────────────────────
    # Step 3：说话人验证特征（v2Pro 专有）
    # ──────────────────────────────────────────────────────────────────

    def _load_sv(self):
        """懒加载 ERes2NetV2 说话人验证模型。"""
        if self._sv_model is not None:
            return self._sv_model

        import kaldi as Kaldi  # type: ignore # noqa
        import torchaudio
        from ERes2NetV2 import ERes2NetV2  # type: ignore # noqa

        logger.info("加载 SV 模型: %s", SV_MODEL_PATH)

        class _SVWrapper:
            def __init__(self, path, device, is_half):
                state = torch.load(path, map_location="cpu")
                # 原ckpt可能只是一系列权重OrderedDict或者包含model_args的dict
                if "model_args" in state:
                    self.model = ERes2NetV2(**state["model_args"])
                    self.model.load_state_dict(state["model"])
                else:
                    self.model = ERes2NetV2(baseWidth=24, scale=4, expansion=4)
                    self.model.load_state_dict(state)
                if is_half:
                    self.model = self.model.half()
                self.model = self.model.to(device).eval()
                self.device = device
                self.is_half = is_half
                self._kaldi = Kaldi

            def get_feature(self, wav_path: str) -> torch.Tensor:
                waveform, sr = torchaudio.load(wav_path)
                if sr != 16000:
                    waveform = torchaudio.functional.resample(
                        waveform, orig_freq=sr, new_freq=16000
                    )

                feat = self._kaldi.fbank(
                    waveform,
                    num_mel_bins=80,
                    sample_frequency=16000,
                    snip_edges=False,
                )
                feat = feat - feat.mean(dim=0, keepdim=True)
                if self.is_half:
                    feat = feat.half()
                with torch.no_grad():
                    emb = self.model(feat.unsqueeze(0).to(self.device))
                return emb.cpu()

        self._sv_model = _SVWrapper(SV_MODEL_PATH, self.device, self.is_half)
        logger.info("SV 模型加载完成")
        return self._sv_model

    def extract_sv_features(self) -> None:
        """
        Step 3：说话人嵌入向量提取（仅 v2Pro/v2ProPlus 需要）。

        使用 ERes2NetV2 为每个 32kHz 重采样音频提取
        说话人 d-vector，保存到 7-sv_cn/。
        """
        if self.version not in ("v2Pro", "v2ProPlus"):
            logger.info("Step 3 跳过（版本 %s 不需要 SV 特征）", self.version)
            return

        if not os.path.isfile(self.list_path):
            raise FileNotFoundError(f"ASR 标注文件不存在: {self.list_path}")

        sv = self._load_sv()

        with open(self.list_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        logger.info("Step 3/4 SV 说话人特征提取 | 共 %d 条", len(lines))

        with tqdm(total=len(lines), desc="Step3 SV", unit="file") as pbar:
            for line in lines:
                try:
                    wav_name = os.path.basename(line.split("|")[0])
                    sv_path = os.path.join(self.sv_dir, f"{wav_name}.pt")
                    wav32_path = os.path.join(self.wav32_dir, wav_name)

                    if os.path.exists(sv_path):
                        pbar.update(1)
                        continue  # 断点续跑

                    if not os.path.exists(wav32_path):
                        logger.warning("32kHz 文件不存在，跳过 SV: %s", wav_name)
                        pbar.update(1)
                        continue

                    emb = sv.get_feature(wav32_path)
                    _my_save(emb, sv_path)

                except Exception:
                    logger.error(
                        "SV 提取失败: %s\n%s", line[:80], traceback.format_exc()
                    )
                pbar.update(1)

        logger.info("Step 3 完成 | SV 特征 -> %s", self.sv_dir)

    # ──────────────────────────────────────────────────────────────────
    # Step 4：语义 Token 量化
    # ──────────────────────────────────────────────────────────────────

    def _load_vq_model(self):
        """
        懒加载 SoVITS 预训练模型的 VQ 量化器。

        仅使用生成器 s2G 的 extract_latent() 接口，
        不需要判别器，不修改权重（eval 模式）。
        """
        if self._vq_model is not None:
            return self._vq_model

        import utils
        from module.models import SynthesizerTrn

        s2g_path = (
            os.path.join(PRETRAINED_DIR, "v2Pro", f"s2G{self.version}.pth")
            if self.version in ("v2Pro", "v2ProPlus")
            else os.path.join(PRETRAINED_DIR, "gsv-v2final-pretrained", "s2G2333k.pth")
        )

        logger.info("加载 SoVITS VQ 模型: %s", s2g_path)
        hps = utils.get_hparams_from_file(S2_CONFIG_PATH)
        vq = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            version=self.version,
            **hps.model,  # type: ignore
        )
        state = torch.load(s2g_path, map_location="cpu", weights_only=False)
        vq.load_state_dict(state["weight"], strict=False)
        if self.is_half:
            vq = vq.half()
        self._vq_model = vq.to(self.device).eval()
        logger.info("VQ 模型加载完成")
        return self._vq_model

    def extract_semantic_tokens(self) -> str:
        """
        Step 4：HuBERT 特征 → 离散语义 Token。

        通过 SoVITS 的 VQ 层将连续 768 维 HuBERT 向量量化为
        整数 token 序列（词汇表约 1024），这是 GPT 训练的直接目标。

        Returns
        -------
        str
            输出文件路径 (6-name2semantic.tsv)
        """
        if not os.path.isfile(self.list_path):
            raise FileNotFoundError(f"ASR 标注文件不存在: {self.list_path}")

        vq = self._load_vq_model()

        with open(self.list_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        logger.info("Step 4/4 语义 Token 量化 | 共 %d 条", len(lines))
        results = []

        with tqdm(total=len(lines), desc="Step4 Semantic", unit="file") as pbar:
            for line in lines:
                try:
                    wav_name = os.path.basename(line.split("|")[0])
                    hubert_path = os.path.join(self.hubert_dir, f"{wav_name}.pt")

                    if not os.path.exists(hubert_path):
                        logger.warning("HuBERT 特征不存在，跳过: %s", wav_name)
                        pbar.update(1)
                        continue

                    ssl = torch.load(hubert_path, map_location="cpu")
                    if self.is_half:
                        ssl = ssl.half()
                    ssl = ssl.to(self.device)

                    # extract_latent: HuBERT (1,768,T) → codes (1,1,T') 离散 token
                    codes = vq.extract_latent(ssl)
                    tokens = " ".join(str(t) for t in codes[0, 0, :].tolist())
                    results.append(f"{wav_name}\t{tokens}")

                except Exception:
                    logger.error(
                        "语义量化失败: %s\n%s", line[:80], traceback.format_exc()
                    )
                pbar.update(1)

        with open(self.semantic_path, "w", encoding="utf-8") as f:
            f.write("\n".join(results))

        logger.info("Step 4 完成 | 语义 Token -> %s", self.semantic_path)
        return self.semantic_path

    # ──────────────────────────────────────────────────────────────────
    # 一键执行全部步骤
    # ──────────────────────────────────────────────────────────────────

    def run_all(self) -> dict[str, str]:
        """
        依次执行全部 4 步特征提取。

        Returns
        -------
        dict
            各步骤输出路径，键名：text / semantic / exp_dir
        """
        logger.info("==== 开始全流程特征提取 | %s ====", self.speaker)

        text_path = self.extract_text_features()
        self.extract_hubert_features()
        self.extract_sv_features()  # v2 时自动跳过
        semantic_path = self.extract_semantic_tokens()

        self.release_models()

        logger.info("==== 特征提取完成 | 实验目录: %s ====", self.exp_dir)
        return {
            "text": text_path,
            "semantic": semantic_path,
            "exp_dir": self.exp_dir,
        }

    def release_models(self) -> None:
        """释放所有已加载模型，节省显存。"""
        self._bert_model = None
        self._bert_tokenizer = None
        self._hubert_model = None
        self._sv_model = None
        self._vq_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("特征提取模型缓存已释放")


if __name__ == "__main__":
    setup_logging()

    # ---- Interactive Inputs ----
    default_speaker = "wangliqun"
    user_speaker = input(
        f"Please enter speaker name (Press Enter to use default '{default_speaker}'): "
    ).strip()
    SPEAKER_NAME = user_speaker if user_speaker else default_speaker

    print("\nSelect Training Model Version:")
    print("1. v2Pro (Recommended)")
    print("2. v2ProPlus")
    version_choice = input("Enter choice (1/2) [Default: 1]: ").strip()

    version_map = {"1": "v2Pro", "2": "v2ProPlus"}
    VERSION = version_map.get(version_choice, "v2Pro")

    print("\nSelect Execution Mode:")
    print("1. run_all (Execute all steps automatically)")
    print("2. step_by_step (Execute step by step with explanations)")
    mode_choice = input("Enter choice (1/2) [Default: 1]: ").strip()
    EXEC_MODE = "step_by_step" if mode_choice == "2" else "run_all"

    logger.info(
        "Initializing FeatureExtractor for speaker: %s, version: %s, mode: %s",
        SPEAKER_NAME,
        VERSION,
        EXEC_MODE,
    )

    extractor = FeatureExtractor(speaker=SPEAKER_NAME, version=VERSION)
    try:
        if EXEC_MODE == "run_all":
            result = extractor.run_all()
            logger.info("Feature extraction results summary:")
            logger.info(" - Experiment Dir: %s", result["exp_dir"])
            logger.info(" - Phoneme Text: %s", result["text"])
            logger.info(" - Semantic Tokens: %s", result["semantic"])
        else:
            logger.info("========== 开启逐步执行与学习模式 ==========")

            logger.info("\n--- 【准备阶段】 ---")
            logger.info(
                "即将执行特征提取。该过程将从ASR标注文件（包含音频路径和对应文本）开始。"
            )
            logger.info(f"输入ASR标注文件所在路径: {extractor.list_path}")
            input("按回车键开始 Step 1 (BERT 文本特征提取)...")

            logger.info("\n--- 【Step 1: BERT 文本特征提取】 ---")
            logger.info("作用：为GPT训练提供当前文本的上下文语义表示。")
            logger.info(
                "原理：使用预训练的中文BERT模型，将标注文本转化为音素，并为每个音素获取嵌入（Embedding）向量。"
            )
            text_path = extractor.extract_text_features()
            logger.info("【Step 1 产生的新文件/输出】:")
            logger.info(f"1. 音素对应文件 (供后续及训练使用): {text_path}")
            logger.info(
                f"2. 每个音频文件对应的 BERT 特征张量 (.pt)，存放在: {extractor.bert_dir}"
            )
            logger.info(
                "解析：这些 .pt 文件将作为GPT微调时的条件输入之一，不需要传递给当前特征提取流程的 Step 2。"
            )

            input("\n按回车键开始 Step 2 (HuBERT 音频语义特征提取)...")
            logger.info("\n--- 【Step 2: HuBERT 音频语义特征提取】 ---")
            logger.info("作用：获取音频的自监督语义表示，作为连接文本和音频的桥梁。")
            logger.info(
                "原理：基于中文 HuBERT 模型，以16kHz输入，提取每一帧（每秒25帧）的特征（768维）。同时，此步骤也会把原始音频统一重采样至32kHz。"
            )
            extractor.extract_hubert_features()
            logger.info("【Step 2 产生的新文件/输出】:")
            logger.info(
                f"1. 每条音频对应的 HuBERT 特征张量 (.pt)，存放于: {extractor.hubert_dir}"
            )
            logger.info(
                f"2. 重采样后的 32kHz 统一音频 (.wav) 存放于: {extractor.wav32_dir}"
            )
            logger.info(
                "解析：生成的 32kHz 音频将会作为 Step 3 (如果是v2Pro+) 的输入，并且在最终的SoVITS训练中作为真值。HuBERT 发出的连续特征向量 (.pt) 将作为 Step 4 的输入！"
            )

            input("\n按回车键开始 Step 3 (说话人验证特征提取 SV)...")
            logger.info("\n--- 【Step 3: 说话人验证特征 (仅v2Pro系列)】 ---")
            logger.info(
                "作用：描述'谁在说话'（音色指纹），用于限制合成语音贴近参考人声。"
            )
            logger.info("原理：使用 ERes2NetV2 提取特征。")
            logger.info(
                f"输入依赖：Step 2 中生成的 32kHz 音频文件 ({extractor.wav32_dir})。"
            )
            extractor.extract_sv_features()
            logger.info("【Step 3 产生的新文件/输出】:")
            logger.info(f"1. 说话人嵌入特征向量 (.pt) 存放在: {extractor.sv_dir}")
            logger.info(
                "解析：这些提取的指纹张量，将会单独喂给后续的 SoVITS 训练作为音色条件变量，与 Step 4 无直接依赖关联。"
            )

            input("\n按回车键开始 Step 4 (语义 Token 量化)...")
            logger.info("\n--- 【Step 4: 语义 Token 量化】 ---")
            logger.info(
                "作用：将连续的音频特征'离散化'，变为类似文本Token的形式（词表约大小1024）。"
            )
            logger.info("原理：使用预训练好的 SoVITS 模型中的 VQ 量化器模块。")
            logger.info(
                f"输入依赖：Step 2 产生的 HuBERT 连续特征文件 (.pt)，目录为: {extractor.hubert_dir}。"
            )
            semantic_path = extractor.extract_semantic_tokens()
            logger.info("【Step 4 产生的新文件/输出】:")
            logger.info(f"1. 汇总所有音频语义Token的列表文件: {semantic_path}")
            logger.info(
                "解析：到此为止，所有数据预处理及特征转化结束。这正是 GPT 模型训练直接预测的目标（从文本预测这些 Token 序列）。"
            )

            logger.info("\n========== 逐步学习与执行结束 ==========")
            logger.info(
                "所有已生成的特征文件（音频、文本、各种 .pt 张量）均全部存放在实验日志目录中，接下来即可将其送入 s1/s2 训练脚本！"
            )

    except Exception:
        logger.error("Feature extraction failed:\n%s", traceback.format_exc())
    finally:
        extractor.release_models()
