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

class GmeTh(threading.Thread):
	"""Last full data: 01/02/2017

	Parameters
	----------
		log : 
			logger to display and save logs

	Attributes
	----------
		name : str
			name of the thread
		log : 
			logger to display and save logs
		spider:
			web crawler collecting data from GME
	
	Methods
	-------
		run()
			starting function. Complete history and daily files download
		getDay()
			downlaod the daily GME files
		getHistory()
			download the full history of GME files
	"""
	def __init__(self, log):
		threading.Thread.__init__(self)
		self.name = 'gmeSpider'
		self.log = log
		self.spider = GMESpider(self.log)
		# Start the thread
		self.start()
		  
	def run(self):
		"""Called when the thread start. 
		If in the configuration files the CREATE_DB flag is set, all the GME 
		available data from 01/02/2017 are downloaded.
		The GME website is daily visited to retrieve the last daily data.
		"""
		self.log.info("Spider Running")
		# Download full history
		if conf.FULL_HISTORY:
			self.getHistory()
			conf.FULL_HISTORY = False
		# Download the daily files
		self.getDay()
			
	def getDay(self):
		"""Visit the GME website and wake up the spider to retrieve 
		the daily data, then send the spider to sleep until the next day.
		"""
		while True:
			date = (datetime.now().strftime('%d/%m/%Y')
			for item in GME:
				self.spider.getData(item, date, date)
			
			self.log.info("Spider waiting...")
			time.sleep(INTER_DATA_GME)
	
	def getHistory(self):
		"""Wake up the spider to download all the full available data from
		01/02/2017 since the current day, then set the spider to sleep.
		"""
		start = datetime(2017, 2, 1)
		limit = datetime.now() + relativedelta(months=+1)
		# Start downloading
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
