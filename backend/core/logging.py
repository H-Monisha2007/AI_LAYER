import logging
import sys
from backend.core.config import settings

def setup_logging():
    log_level = logging.DEBUG if settings.DEBUG else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("trust_ai")
    logger.info(f"Initialized logging for {settings.APP_NAME} in {settings.APP_ENV} mode")
    return logger

logger = setup_logging()
