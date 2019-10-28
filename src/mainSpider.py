import logging
import logging.config
from spiders.gme import *
from common.config import *
from spiders.processor import *

logging.config.fileConfig('common/logging.conf')
logger = logging.getLogger(__name__)

# Create the processor and the GME spider
gme_processor = ProcessorTh(logger)
gme_sp = GMESpider(logger)
# Start the spider
gme_sp.getData(GME[0], '15/10/2019', '25/10/2019')
gme_sp.getData(GME[1], '15/10/2019', '25/10/2019')
