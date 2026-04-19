"""
Logger 模块
- 按天滚动（TimedRotatingFileHandler）
- 单文件超过 20MB 自动新建（大小滚动）
- 同时输出到文件和 console
- 格式类似 Java logback: 时间 [级别] 文件名:行号 - 内容

滚动文件命名规则:
    application.log                  ← 当前写入
    application.log.2026-02-23       ← 当天第一次滚动（按大小 or 跨天）
    application.log.2026-02-23.1     ← 当天第二次
    application.log.2026-02-23.2     ← 当天第三次
"""

import logging
import os
import time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
#  可在此处修改日志配置
# ------------------------------------------------------------------ #
LOG_FILE_NAME = "application.log"  # 日志文件名
LOG_MAX_BYTES = 20 * 1024 * 1024  # 单文件最大 20 MB
LOG_BACKUP_COUNT = 30  # 保留最近 30 天（或 30 个滚动文件）
LOG_LEVEL = logging.DEBUG  # 全局最低级别
LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d [%(levelname)-8s] %(filename)s:%(lineno)d - %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
# ------------------------------------------------------------------ #


class _SizeAndTimeRotatingHandler(TimedRotatingFileHandler):
    """
    在 TimedRotatingFileHandler（按天滚动）基础上叠加大小限制。

    性能说明
    --------
    emit() 中使用 self.stream.tell() 获取当前写入偏移量：
        - tell() 读取的是 OS 文件描述符中缓存的偏移量，属于纯内存操作
        - 无磁盘 I/O，单次耗时约 50~200 ns，对吞吐量影响可忽略
        - 对比：os.path.getsize() 需要 stat() syscall，慢 10~100 倍
    """

    def __init__(self, filename: str, max_bytes: int = LOG_MAX_BYTES, **kwargs):
        super().__init__(filename, **kwargs)
        self.max_bytes = max_bytes

    # ---------- 核心：大小 + 时间双触发 ---------- #
    def emit(self, record: logging.LogRecord) -> None:
        """
        写日志前检查文件大小，超过阈值则先滚动再写入。
        self.stream.tell() 是内存操作，性能极低。
        """
        try:
            if self.stream and self.stream.tell() >= self.max_bytes:
                self.doRollover()
        except Exception:
            pass  # 滚动失败不影响日志写入
        super().emit(record)

    # ---------- 滚动：处理同天多次滚动的文件命名 ---------- #
    def doRollover(self) -> None:
        """
        覆写滚动逻辑，支持同天多次滚动时在文件名后附加序号。

        时间滚动（跨天）：application.log → application.log.2026-02-23
        大小滚动（同天）：application.log → application.log.2026-02-23.1 / .2 ...
        """
        # 1. 关闭当前流
        if self.stream:
            self.stream.close()
            self.stream = None  # type: ignore[assignment]

        # 2. 计算目标文件名（时间后缀）
        current_time = int(time.time())
        # rolloverAt 是下一次时间滚动的时间戳，减去 interval 得到本周期开始时间
        time_tuple = time.localtime(self.rolloverAt - self.interval)
        time_suffix = time.strftime(self.suffix, time_tuple)
        base_name = (
            self.baseFilename + "." + time_suffix
        )  # e.g. application.log.2026-02-23

        # 3. 若同名文件已存在，追加序号
        target_name = base_name
        seq = 1
        while os.path.exists(target_name):
            target_name = f"{base_name}.{seq}"
            seq += 1

        # 4. 重命名当前日志文件
        if os.path.exists(self.baseFilename):
            os.rename(self.baseFilename, target_name)

        # 5. 删除过期备份（超出 backupCount）
        self._delete_old_backups()

        # 6. 重新打开新日志文件
        self.stream = self._open()

        # 7. 仅当真正跨天时才更新下次滚动时间
        if current_time >= self.rolloverAt:
            self.rolloverAt = self.computeRollover(current_time)

    def _delete_old_backups(self) -> None:
        """删除超出 backupCount 的旧备份文件（按修改时间排序）。"""
        log_dir = Path(self.baseFilename).parent
        base = Path(self.baseFilename).name
        # 匹配所有滚动文件：application.log.YYYY-MM-DD 或 application.log.YYYY-MM-DD.N
        backups = sorted(
            log_dir.glob(f"{base}.*"),
            key=lambda p: p.stat().st_mtime,
        )
        # 保留最新的 backupCount 个，多余的删除
        for old_file in backups[: max(0, len(backups) - self.backupCount)]:
            old_file.unlink(missing_ok=True)


# ------------------------------------------------------------------ #
#  库根节点名称（所有子模块 logger 的公共父节点）
# ------------------------------------------------------------------ #
_LIB_ROOT = "server"

# 库初始化时，在根节点挂一个 NullHandler。
# 这是 Python 官方对"库"的强制要求：
#   - NullHandler 防止"No handlers could be found"警告
#   - 不配置任何真实 handler，避免污染使用方的日志系统
logging.getLogger(_LIB_ROOT).addHandler(logging.NullHandler())


# ------------------------------------------------------------------ #
#  setup_logging：供使用方（外部项目）在启动时调用一次
# ------------------------------------------------------------------ #
def setup_logging(
    level: int = LOG_LEVEL,
    log_dir: Path | None = None,
    log_file_name: str = LOG_FILE_NAME,
    max_bytes: int = LOG_MAX_BYTES,
    backup_count: int = LOG_BACKUP_COUNT,
    enable_console: bool = True,
    extra_loggers: list[str] | None = None,
) -> None:
    # formater
    default_formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )
    # logger directory
    actual_log_dir = log_dir if log_dir is not None else LOGS_DIR
    actual_log_dir = Path(actual_log_dir)
    actual_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = actual_log_dir / log_file_name
    # handler
    file_handler = _SizeAndTimeRotatingHandler(
        filename=str(log_file),
        max_bytes=max_bytes,
        when="midnight",
        interval=1,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(default_formatter)
    handlers: list[logging.Handler] = [file_handler]

    # ---- Console Handler ---- #
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        handlers.append(console_handler)

    # logger configuration
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    for handler in handlers:
        handler.setFormatter(default_formatter)
        root_logger.addHandler(handler)

    core_loggers = [
        _LIB_ROOT,
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
    ]

    if extra_loggers:
        core_loggers.extend(extra_loggers)

    for logger_root in core_loggers:
        logger = logging.getLogger(logger_root)
        logger.handlers.clear()
        logger.setLevel(level)
        for handler in handlers:
            logger.addHandler(handler)
        logger.propagate = False

    logging.getLogger("watchfiles").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("langsmith").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.INFO)
    logging.getLogger("requests").setLevel(logging.INFO)
    logging.getLogger("huggingface_hub").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.INFO)

    root_logger.info("日志系统初始化完成（统一接管 uvicorn + root）")
