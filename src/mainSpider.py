import sys
import logging
import logging.config
import src.common.config as conf
from src.loggerbot import bot
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
from src.spiders.gme import *
from src.common.config import *

sys.dont_write_bytecode = True

logging.config.fileConfig('src/common/logging.conf')
logger = logging.getLogger(__name__)

def getDay():
	"""Visit the GME website and wake up the spider to retrieve 
	the daily data, then send the spider to sleep until the next day.
	"""
	bot('INFO', 'GME', 'getDaily started.')
	spider = GMESpider(logger)
	date = datetime.now().strftime('%d/%m/%Y')
	date_nxt = (datetime.now() + relativedelta(days=+1)).strftime('%d/%m/%Y')
	for item in GME:
		spider.getData(item, date, date)
	for item in GME_NEXT:
		spider.getData(item, date_nxt, date_nxt)
	spider.driver.quit()
	bot('INFO', 'GME', 'getDaily ended.')

def getHistory():
	"""Wake up the spider to download all the full available data from
	01/02/2017 since the current day, then set the spider to sleep.
	"""
	spider = GMESpider(logger)
	start = datetime(2017, 2, 1)
	limit = datetime.now() + relativedelta(months=+1)
	# Start downloading
	while start.strftime('%m/%Y') != limit.strftime('%m/%Y'):
		end = start + relativedelta(months=+1, days=-1)
		
		for item in GME:
			spider.getData(
				item, 
				start.strftime('%d/%m/%Y'), 
				end.strftime('%d/%m/%Y')
		)
		
		start = end + relativedelta(days=+ 1)