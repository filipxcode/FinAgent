import logging

from src.config.config import LoggingSettings

_configured = False


def configure_logging(settings: LoggingSettings | None = None) -> None:
    """Configure the root logger once.

    Every module logger (logging.getLogger(__name__), e.g. in src.flow.*)
    propagates up to root by default, so configuring root here is enough to
    make debug/info logs show up everywhere without per-module setup.
    Calling this more than once is a no-op.
    """
    global _configured
    if _configured:
        return

    settings = settings or LoggingSettings()
    logging.basicConfig(level=settings.level, format=settings.format)
    _configured = True
