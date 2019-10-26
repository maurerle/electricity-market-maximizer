import logging
import logging.config
from src.spiders import mainSpider

logging.config.fileConfig('src/common/logging.conf')
logger = logging.getLogger(__name__)



