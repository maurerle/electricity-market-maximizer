import logging
import logging.config
import threading
from datetime import datetime, timedelta
from spiders.gme import *
from common.config import *
from spiders.processor import *
"""Last data 01/02/2017"""

class GmeTh(threading.Thread):
	def __init__(self, log):
		threading.Thread.__init__(self)
		self.name = 'gmeSpider'
		self.log = log
		
		self.start()
		  
	def run(self):
		self.log.info("Spider Running")
		spider = GMESpider(self.log)
		date = (datetime.now() - timedelta(1)).strftime('%d/%m/%Y')
		while True:
			for item in GME:
				spider.getData(item, date, date)
			self.log.info("Spider waiting...")
			time.sleep(INTER_DATA_GME)
	

logging.config.fileConfig('common/logging.conf')
logger = logging.getLogger(__name__)

# Create the processor and the GME spider
gme_processor = ProcessorTh(logger)
gme_spider = GmeTh(logger)
