import sys
import logging
import logging.config
from src.loggerbot import bot
from src.database.processor import FileProcessor
from dateutil.relativedelta import relativedelta
from datetime import datetime
from src.spiders.gme import *
from src.spiders.terna import *
from src.common.config import *

sys.dont_write_bytecode = True

logging.config.fileConfig('src/common/logging.conf')
logger = logging.getLogger(__name__)

def getDay():
	"""Visit the GME website and wake up the spider to retrieve 
	the daily data, then send the spider to sleep until the next day.
	"""
	bot('INFO', 'GME', 'getDaily started.')
	gme = GMESpider(logger)
	terna = TernaSpider()
	processor = FileProcessor(logger)
	
	date = datetime.now().strftime('%d/%m/%Y')
	date_nxt = (datetime.now() + relativedelta(days=+1)).strftime('%d/%m/%Y')
	date_week = (datetime.now() + relativedelta(days=-8)).strftime('%d/%m/%Y')

	terna(TERNA['url'], date, date)
	terna.driver.quit()
	for item in GME:
		gme.getData(item, date, date)
	for item in GME_NEXT:
		gme.getData(item, date_nxt, date_nxt)
	gme.getData(GME_WEEK, date_week)

	gme.driver.quit()
	processor.stop()
	bot('INFO', 'GME', 'getDaily ended.')

def getHistory():
	"""Wake up the spider to download all the full available data from
	01/02/2017 since the current day, then set the spider to sleep.
	"""
	bot('INFO', 'GME', 'getHistory started.')
	gme = GMESpider(logger)
	processor = FileProcessor(logger)

	start = START
	limit = datetime.now()
	
	# Start downloading
	while start.strftime('%m/%Y') != limit.strftime('%m/%Y'):
		end = start + relativedelta(months=+1, days=-1)
		
		for item in GME:
			gme.getData(
				item, 
				start.strftime('%d/%m/%Y'), 
				end.strftime('%d/%m/%Y')
			)
		for item in GME_NEXT:
			gme.getData(
				item, 
				start.strftime('%d/%m/%Y'), 
				end.strftime('%d/%m/%Y')
			)
		
		start = end + relativedelta(days=+ 1)
	
	# Get the current month until the day before the current
	end = datetime.now()+ relativedelta(days=-1)
	
	for item in GME:
		gme.getData(
			item, 
			start.strftime('%d/%m/%Y'), 
			end.strftime('%d/%m/%Y')
		)
	for item in GME_NEXT:
		gme.getData(
				item, 
				start.strftime('%d/%m/%Y'), 
				datetime.now().strftime('%d/%m/%Y')
		)
	
	# Get the public offers referred to one week before the current
	start = START
	limit = datetime.now() + relativedelta(days=-7)
	
	while start.strftime('%dd/%mm/%YY') != limit.strftime('%dd/%mm/%YY'):
		gme.getData(GME_WEEK, start.strftime('%d/%m/%Y'))
		start += relativedelta(days=+1)

	gme.driver.quit()
	processor.stop()
	bot('INFO', 'GME', 'getHistory ended.')