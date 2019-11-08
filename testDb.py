import logging
import logging.config
from src.database.processor import FileProcessor

logging.config.fileConfig('src/common/logging.conf')
logger = logging.getLogger(__name__)
FileProcessor(logger)