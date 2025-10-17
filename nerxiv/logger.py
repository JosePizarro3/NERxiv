import copy
import logging
import os
from pathlib import Path


try:
    import structlog
except Exception:  # pragma: no cover - optional runtime dependency
    structlog = None

# List to store log messages
log_storage = []


def store_log_message(_, __, event_dict):
    """
    Custom processor to store log messages in a list of dictionaries containing the log messages as:

        {
            'event': <the log message>,
            'timestamp': <the timestamp>,
            'level': <the log level (info, debug, warning, etc)>,
        }
    """
    log_storage.append(copy.deepcopy(event_dict))
    return event_dict


log_to_file = os.getenv("PYRXIV_LOG_TO_FILE", "1") == "1"
log_path = Path("./data/logs.json")

if log_to_file and log_path.parent.exists():
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        filename=log_path,
        filemode="w",
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

# Add this basic config to ensure logs go to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    # stream=sys.stdout,
    filename="./data/logs.json",
    filemode="w",
)


if structlog is not None:
    # Configure structlog with the custom processor
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.CallsiteParameterAdder(
                [
                    structlog.processors.CallsiteParameter.PATHNAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
            store_log_message,
            # structlog.dev.ConsoleRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),  # Use stdlib logger backend
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
    )

    # Create a structlog logger instance
    logger = structlog.get_logger()
else:
    # Fallback plain stdlib logger when structlog isn't installed. Keep the
    # `store_log_message` processor behavior by attaching a simple wrapper
    # that forwards log records to the in-memory `log_storage`.
    logger = logging.getLogger("nerxiv")
    logger.setLevel(logging.INFO)

    # ensure we have at least one handler to avoid "No handler" warnings
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    # provide convenience methods that also store messages in log_storage
    def _store_and_log(level_fn, level_name):
        def _fn(message, *args, **kwargs):
            # emulate a minimal event_dict stored by structlog processor
            event = {"event": str(message), "level": level_name}
            log_storage.append(event)
            level_fn(message, *args, **kwargs)

        return _fn

    logger.info = _store_and_log(logger.info, "info")
    logger.debug = _store_and_log(logger.debug, "debug")
    logger.warning = _store_and_log(logger.warning, "warning")
    logger.error = _store_and_log(logger.error, "error")
    logger.exception = _store_and_log(logger.exception, "exception")


import functools
import warnings


def deprecated(message="This function is deprecated."):
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapped

    return decorator
