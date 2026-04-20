"""
Logger Module
- Daily rotation (TimedRotatingFileHandler)
- Size rotation (auto-creates new file when > 20MB)
- Output to both file and console
- Format similar to Java Logback: Time [Level] File:Line - Message

Rotation naming rules:
    application.log                  ← Current active log
    application.log.2026-04-20       ← First rotation of the day
    application.log.2026-04-20.1     ← Second rotation
    application.log.2026-04-20.2     ← Third rotation
"""

import logging
import os
import time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
#  Adjust Logging Configurations Here
# ------------------------------------------------------------------ #
LOG_FILE_NAME = "application.log"  # Log filename
LOG_MAX_BYTES = 20 * 1024 * 1024  # Max size per file (20 MB)
LOG_BACKUP_COUNT = 30  # Retention: 30 days or 30 rotation files
LOG_LEVEL = logging.DEBUG  # Global minimum log level
LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d [%(levelname)-8s] %(filename)s:%(lineno)d - %(message)s"
)
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
# ------------------------------------------------------------------ #


class _SizeAndTimeRotatingHandler(TimedRotatingFileHandler):
    """
    Overlays size limits on top of TimedRotatingFileHandler (daily rotation).

    Performance Note:
    ----------------
    Uses self.stream.tell() to get current write offset in emit():
        - tell() reads the offset from OS file descriptor cache (memory operation).
        - No disk I/O, cost is ~50-200 ns per call, negligible impact on throughput.
        - Contrast: os.path.getsize() requires stat() syscall, 10x~100x slower.
    """

    def __init__(self, filename: str, max_bytes: int = LOG_MAX_BYTES, **kwargs):
        super().__init__(filename, **kwargs)
        self.max_bytes = max_bytes

    # ---------- Core: Dual Trigger (Size + Time) ---------- #
    def emit(self, record: logging.LogRecord) -> None:
        """
        Check file size before writing; rollover if threshold exceeded.
        """
        try:
            if self.stream and self.stream.tell() >= self.max_bytes:
                self.doRollover()
        except Exception:
            pass  # Rollover failure should not stop logging
        super().emit(record)

    # ---------- Rollover: Handle multiple rotations in a single day ---------- #
    def doRollover(self) -> None:
        """
        Overrides rollover logic to append sequence numbers for multiple rotations per day.

        Time-based (Day cross): application.log → application.log.2026-04-20
        Size-based (Same day):  application.log → application.log.2026-04-20.1 / .2 ...
        """
        # 1. Close current stream
        if self.stream:
            self.stream.close()
            self.stream = None  # type: ignore[assignment]

        # 2. Calculate target filename (time suffix)
        current_time = int(time.time())
        # rolloverAt is the timestamp for next time-based rotation
        time_tuple = time.localtime(self.rolloverAt - self.interval)
        time_suffix = time.strftime(self.suffix, time_tuple)
        base_name = (
            self.baseFilename + "." + time_suffix
        )  # e.g., application.log.2026-04-20

        # 3. If file exists, append sequence number
        target_name = base_name
        seq = 1
        while os.path.exists(target_name):
            target_name = f"{base_name}.{seq}"
            seq += 1

        # 4. Rename current log file
        if os.path.exists(self.baseFilename):
            os.rename(self.baseFilename, target_name)

        # 5. Delete old backups exceeding backupCount
        self._delete_old_backups()

        # 6. Open new stream
        self.stream = self._open()

        # 7. Update next rollover time only if a day actually crossed
        if current_time >= self.rolloverAt:
            self.rolloverAt = self.computeRollover(current_time)

    def _delete_old_backups(self) -> None:
        """Deletes old backups exceeding backupCount (sorted by modified time)."""
        log_dir = Path(self.baseFilename).parent
        base = Path(self.baseFilename).name
        # Match application.log.YYYY-MM-DD or application.log.YYYY-MM-DD.N
        backups = sorted(
            log_dir.glob(f"{base}.*"),
            key=lambda p: p.stat().st_mtime,
        )
        # Keep latest backupCount files
        for old_file in backups[: max(0, len(backups) - self.backupCount)]:
            old_file.unlink(missing_ok=True)


# ------------------------------------------------------------------ #
#  Library Root Name (Parent for all sub-module loggers)
# ------------------------------------------------------------------ #
_LIB_ROOT = "server"

# Attach NullHandler to root on init as per official Python guidelines
logging.getLogger(_LIB_ROOT).addHandler(logging.NullHandler())


# ------------------------------------------------------------------ #
#  setup_logging: Called once by entry point
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
    # Formatter
    default_formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )
    # Logger directory
    actual_log_dir = log_dir if log_dir is not None else LOGS_DIR
    actual_log_dir = Path(actual_log_dir)
    actual_log_dir.mkdir(parents=True, exist_ok=True)
    log_file = actual_log_dir / log_file_name

    # File Handler
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

    # Console Handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        handlers.append(console_handler)

    # Root logger configuration
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

    # External library levels
    logging.getLogger("watchfiles").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("langsmith").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.INFO)
    logging.getLogger("requests").setLevel(logging.INFO)
    logging.getLogger("huggingface_hub").setLevel(logging.INFO)
    logging.getLogger("transformers").setLevel(logging.INFO)

    root_logger.info("Logging system initialized (unified handlers for uvicorn + root)")
