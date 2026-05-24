from __future__ import annotations

import logging
import sys


def _force_utf8(stream) -> None:
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

_LEVEL_COLORS = {
    "DEBUG": "\x1b[36m",      # cyan
    "INFO": "\x1b[32m",       # green
    "WARNING": "\x1b[33m",    # yellow
    "ERROR": "\x1b[31m",      # red
    "CRITICAL": "\x1b[1;31m", # bold red
}
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelname, "")
        ts = f"{_DIM}{self.formatTime(record, self.datefmt)}.{int(record.msecs):03d}{_RESET}"
        lvl = f"{color}{record.levelname:<7}{_RESET}"
        name = f"{_BOLD}{record.name:<18}{_RESET}"
        line = f"{ts}  {lvl}  {name}  {record.getMessage()}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        if record.stack_info:
            line += "\n" + self.formatStack(record.stack_info)
        return line


def setup_logging(level: int = logging.INFO, log_file: str | None = None, log_file_level: int | None = None) -> None:
    _force_utf8(sys.stdout)
    _force_utf8(sys.stderr)
    console = logging.StreamHandler()
    console.setFormatter(ColorFormatter(datefmt="%H:%M:%S"))
    console.setLevel(level)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(console)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)-18s  %(message)s",
            datefmt="%H:%M:%S",
        ))
        file_handler.setLevel(log_file_level if log_file_level is not None else level)
        root.addHandler(file_handler)
    root.setLevel(min(level, log_file_level) if log_file and log_file_level is not None else level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.INFO)


def uvicorn_log_config() -> dict:
    _force_utf8(sys.stdout)
    _force_utf8(sys.stderr)
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": ColorFormatter,
                "datefmt": "%H:%M:%S",
            },
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "": {"handlers": ["default"], "level": "INFO"},
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }
