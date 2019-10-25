import logging
import logging.config

logging.config.fileConfig('src/logging.conf')
logger = logging.getLogger(__name__)

