"""
模型微调/训练模块 —— GPT-SoVITS 训练核心（Stage 1 & Stage 2）

【为什么需要分两步训练？】
人类的语音包含两部分信息：
1. 韵律（怎么说）：重音、停顿、语气、语速。
2. 音色（谁在说）：声线特征、频段特点。

GPT-SoVITS 采用解耦设计的核心思想：
- **GPT 模型**（负责“怎么说”）：将文本（文字+BERT向量）翻译成包含韵律的“语义 Token”。它是一个自回归模型，像 ChatGPT 写文章一样，逐个预测下一个发音的 Token。
- **SoVITS 模型**（负责“谁在说”）：将上一步的“语义 Token”，结合参考音频的声纹（SV 特征）和梅尔频谱，还原成最终的波形声音。

【输入数据依赖】
本模块依赖 `feature_extractor.py` 生成的以下特征：
- 2-name2text.txt (文本与音素)
- 3-bert/*.pt (BERT 语义)
- 4-cnhubert/*.pt (HuBERT 连续语义)
- 5-wav32k/*.wav (高保真目标音频)
- 6-name2semantic.tsv (离散语义 Token)
- 7-sv_cn/*.pt (声纹指纹，v2Pro需要)
"""

import json
import logging
import os
from pathlib import Path

import yaml

from server.config import ASSETS_DIR, MODELS_DIR, PRETRAINED_DIR, get_speaker_logs_dir
from server.logger import setup_logging

logger = logging.getLogger("server.trainer")

_project_root = str(Path(__file__).parent.parent.absolute())
_gpt_sovits_dir = os.path.join(_project_root, "GPT_SoVITS")


class SpeakerTrainer:
    def __init__(self, speaker: str, version: str = "v2Pro"):
        self.speaker = speaker
        self.version = version
        self.exp_dir = get_speaker_logs_dir(speaker)
        # 获取 ASSETS 目录下对应的说话人目录
        self.speaker_assets_dir = os.path.join(ASSETS_DIR, speaker)
        os.makedirs(self.speaker_assets_dir, exist_ok=True)

        # 设定训练输出目录: models/speaker/{speaker_name}/
        self.model_out_dir = os.path.join(MODELS_DIR, "speaker", speaker)
        os.makedirs(self.model_out_dir, exist_ok=True)

        # 预训练模型的路径，注意由于文件结构原因目前v2Pro/v2ProPlus的预训练模型都放在 v2Pro 文件夹
        self.s1_pretrained_dir = (
            os.path.join(PRETRAINED_DIR, "v2Pro")
            if version in ("v2Pro", "v2ProPlus")
            else os.path.join(PRETRAINED_DIR, version)
        )
        self.s2_pretrained_dir = (
            os.path.join(PRETRAINED_DIR, "v2Pro")
            if version in ("v2Pro", "v2ProPlus")
            else os.path.join(PRETRAINED_DIR, version)
        )

        # 将生成的配置文件放置于对应的 ASSETS 说话人根目录下
        self.s1_config_path = os.path.join(self.speaker_assets_dir, "s1_train.yaml")
        self.s2_config_path = os.path.join(self.speaker_assets_dir, "s2_train.json")

    def prepare_sovits_config(self, batch_size: int = 8, epochs: int = 8):
        """
        准备 SoVITS (声学模型) 训练配置

        【逻辑意义与原理】
        在这个阶段，SoVITS 模型学习如何把抽象的 Semantic Token 逆向转化为具有特定音色（speaker 音色）的波形。
        它类似于一个"神经声码器 + 变声器"。
        - 目标函数（Loss）：不仅有波形的 L1 损失，还有梅尔频谱的损失，以及对抗生成网络（GAN）的 Discriminator 损失，用来保证声音的真实度和高频细节。
        """
        logger.info(">>> 开始准备 SoVITS (Stage 2) 训练配置...")
        base_config_name = (
            f"s2{self.version}.json"
            if self.version in ("v2Pro", "v2ProPlus")
            else "s2.json"
        )
        base_config_file = os.path.join(_gpt_sovits_dir, "configs", base_config_name)

        with open(base_config_file, "r", encoding="utf-8") as f:
            s2_config = json.load(f)

        # 修改配置以指向我们生成的特征目录
        s2_config["train"]["batch_size"] = batch_size
        s2_config["train"]["epochs"] = epochs
        s2_config["train"]["exp_name"] = self.speaker
        # 兼容 s2_train.py 的启动逻辑，默认指定单卡 0号卡
        if "gpu_numbers" not in s2_config["train"]:
            s2_config["train"]["gpu_numbers"] = "0"

        if "save_every_epoch" not in s2_config["train"]:
            s2_config["train"]["save_every_epoch"] = 4
        if "if_save_latest" not in s2_config["train"]:
            s2_config["train"]["if_save_latest"] = 1
        if "if_save_every_weights" not in s2_config["train"]:
            s2_config["train"]["if_save_every_weights"] = True

        # 保存时使用名称，避免 AttributeError: 'HParams' object has no attribute 'name'
        s2_config["name"] = self.speaker

        # 默认填入对应的底模权重文件路径
        s2_config["train"]["pretrained_s2G"] = os.path.join(
            self.s2_pretrained_dir, f"s2G{self.version}.pth"
        )
        s2_config["train"]["pretrained_s2D"] = os.path.join(
            self.s2_pretrained_dir, f"s2D{self.version}.pth"
        )

        # 设置训练输出目录，放到 models/speaker/音色人/SoVITS_weights
        s2_config["train"]["save_dir"] = os.path.join(
            self.model_out_dir, "SoVITS_weights"
        )
        os.makedirs(s2_config["train"]["save_dir"], exist_ok=True)

        # 数据集路径对齐
        s2_config["data"]["exp_dir"] = self.exp_dir

        # 兼容最新版本的 version 字段
        if "model" not in s2_config:
            s2_config["model"] = {}
        s2_config["model"]["version"] = self.version

        # 保存特定于说话人的配置文件
        with open(self.s2_config_path, "w", encoding="utf-8") as f:
            json.dump(s2_config, f, indent=4, ensure_ascii=False)

        logger.info(f"SoVITS 配置文件已生成: {self.s2_config_path}")
        return self.s2_config_path

    def prepare_gpt_config(self, batch_size: int = 8, epochs: int = 15):
        """
        准备 GPT (语言和韵律模型) 训练配置

        【逻辑意义与原理】
        该模型是一个 Transformer 架构。它将“输入文本的 BERT 向量”作为提示词（Prompt），预测序列化的 Semantic Token。
        - 因为在推理时，我们只有文字，没有包含韵律的特征，所以必须靠 GPT 的想象力，去“猜”这句话用什么样的语调、停顿来阅读最合适。
        - 目标函数（Loss）：典型的交叉熵损失（Cross Entropy Loss），像训练语言模型一样训练音频 Token 的预测能力。
        """
        logger.info(">>> 开始准备 GPT (Stage 1) 训练配置...")
        # 默认的 GPT 配置文件
        base_config_name = "s1longer.yaml"  # GPT-SoVITS 通用的 s1 配置基础
        base_config_file = os.path.join(_gpt_sovits_dir, "configs", base_config_name)

        with open(base_config_file, "r", encoding="utf-8") as f:
            s1_config = yaml.safe_load(f)

        # 设置输入数据的路径
        s1_config["train"]["batch_size"] = batch_size
        s1_config["train"]["epochs"] = epochs
        s1_config["train"]["exp_name"] = self.speaker
        if "gpu_numbers" not in s1_config["train"]:
            s1_config["train"]["gpu_numbers"] = "0"

        if "save_every_n_epoch" not in s1_config["train"]:
            s1_config["train"]["save_every_n_epoch"] = 5
        if "if_save_latest" not in s1_config["train"]:
            s1_config["train"]["if_save_latest"] = 1
        if "if_save_every_weights" not in s1_config["train"]:
            s1_config["train"]["if_save_every_weights"] = True

        # 兼容最新版保存名
        s1_config["name"] = self.speaker

        # 设置 GPT (s1) 预训练底模所在路径，兼容训练时未指定报错
        # v2Pro 和 v2ProPlus 通常使用的是此固定底模
        s1_config["train"]["pretrained_s1"] = os.path.join(
            PRETRAINED_DIR,
            "gsv-v2final-pretrained",
            "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        )

        s1_config["train"]["save_dir"] = os.path.join(self.model_out_dir, "GPT_weights")
        os.makedirs(s1_config["train"]["save_dir"], exist_ok=True)

        s1_config["data"]["text_dir"] = os.path.join(self.exp_dir, "2-name2text.txt")
        s1_config["data"]["phones_dir"] = os.path.join(
            self.exp_dir, "6-name2semantic.tsv"
        )
        s1_config["data"]["bert_dir"] = os.path.join(self.exp_dir, "3-bert")

        # 兼容 s1_train.py 的 version 字段
        if "model" not in s1_config:
            s1_config["model"] = {}
        s1_config["model"]["version"] = self.version

        with open(self.s1_config_path, "w", encoding="utf-8") as f:
            yaml.dump(s1_config, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"GPT 配置文件已生成: {self.s1_config_path}")
        return self.s1_config_path

    def print_training_commands(self):
        """
        打印具体的启动指令。在实际生产或者流水线中，可以使用 subprocess 去拉起。
        """
        logger.info("\n" + "=" * 50)
        logger.info("配置准备完毕！请参考以下命令分别启动 SoVITS 和 GPT 的微调训练：")
        logger.info("=" * 50)

        sovits_cmd = f"uv run GPT_SoVITS/s2_train.py --config {self.s2_config_path}"
        gpt_cmd = f"uv run GPT_SoVITS/s1_train.py --config_file {self.s1_config_path}"

        logger.info("\n【1. 训练 SoVITS 声学模型】")
        logger.info(
            "作用：学习该说话人的音色，建立从[Token -> 声纹 -> 波形]的映射网络。"
        )
        logger.info(f"执行终端执行：\n    {sovits_cmd}\n")

        logger.info("【2. 训练 GPT 语言/韵律模型】")
        logger.info("作用：学习该说话人的发音节奏、停顿习惯、语气起伏。")
        logger.info(f"执行终端执行：\n    {gpt_cmd}\n")
        logger.info("=" * 50)


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

    logger.info(
        "Initializing SpeakerTrainer for speaker: %s, version: %s",
        SPEAKER_NAME,
        VERSION,
    )

    trainer = SpeakerTrainer(speaker=SPEAKER_NAME, version=VERSION)
    trainer.prepare_sovits_config()
    trainer.prepare_gpt_config()
    trainer.print_training_commands()
