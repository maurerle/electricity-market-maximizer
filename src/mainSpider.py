import sys
import logging
import logging.config
import threading
from datetime import datetime, timedelta
from spiders.gme import *
from common.config import *
import common.config as conf
from spiders.processor import *
from dateutil.relativedelta import relativedelta

sys.dont_write_bytecode = True

"""Last data 01/02/2017"""

class GmeTh(threading.Thread):
	def __init__(self, log):
		threading.Thread.__init__(self)
		self.name = 'gmeSpider'
		self.log = log
		self.spider = GMESpider(self.log)
		self.start()
		  
	def run(self):
		self.log.info("Spider Running")
		
		if conf.CREATE_DB:
			self.getHistory()
			conf.CREATE_DB = False
		
		self.getDay()
			
	def getDay(self):
		while True:
			date = (datetime.now() - timedelta(1)).strftime('%d/%m/%Y')
			for item in GME:
				self.spider.getData(item, date, date)
			self.log.info("Spider waiting...")
			time.sleep(INTER_DATA_GME)
	
	def getHistory(self):
		start = datetime(2017, 2, 1)
		limit = datetime.now() + relativedelta(months=+1)
		while start.strftime('%m/%Y') != limit.strftime('%m/%Y'):
			end = start + relativedelta(months=+1, days=-1)
			for item in GME:
				self.spider.getData(
					item, 
					start.strftime('%d/%m/%Y'), 
					end.strftime('%d/%m/%Y')
			)
			
			start = end + relativedelta(days=+ 1)

logging.config.fileConfig('common/logging.conf')
logger = logging.getLogger(__name__)

# Create the processor and the GME spider
gme_processor = ProcessorTh(logger)
gme_spider = GmeTh(logger)
