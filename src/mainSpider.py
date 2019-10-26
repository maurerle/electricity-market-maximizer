import logging
import logging.config
from spiders.gme import *
from common.config import *

logging.config.fileConfig('common/logging.conf')
logger = logging.getLogger(__name__)

gme_sp = GMESpider(logger)
gme_sp.getData(GME[0], '15/10/2019', '25/10/2019')
gme_sp.getData(GME[1], '15/10/2019', '25/10/2019')
