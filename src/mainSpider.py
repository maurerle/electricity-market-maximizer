from sys import dont_write_bytecode
import logging
import logging.config
from src.loggerbot.bot import bot
from src.database.processor import FileProcessor
from dateutil.relativedelta import relativedelta
from datetime import datetime
from src.spiders.gme import GMESpider
from src.spiders.terna import TernaSpider, TernaReserve
from src.common.config import GME, GME_NEXT, GME_WEEK, START, TERNA

dont_write_bytecode = True

logging.config.fileConfig('src/common/logging.conf')
logger = logging.getLogger(__name__)

def getDay():
	"""Visit the GME website and Terna one, call the spiders to retrieve 
	the daily data and call the file processor to update the file to a MongoDB
	database.
	"""
	bot('INFO', 'GME/TERNA/TERNA2', 'getDaily started.')
	
	# Classes init
	processor = FileProcessor(logger)
	
	# Date creation
	date = datetime.now().strftime('%d/%m/%Y')
	date_nxt = (datetime.now() + relativedelta(days=+1)).strftime('%d/%m/%Y')
	date_week = (datetime.now() + relativedelta(days=-7)).strftime('%d/%m/%Y')
	
	#=====================================
	# Terna Secondary Reserve spider works
	#=====================================
	logger.info('[TERNA2] getDay() started.')
	
	terna = TernaReserve()
	terna.getDaily()
	terna.driver.quit()
	
	# Logs 
	bot('INFO', 'TERNA2', 'getDaily ended.')
	logger.info('[TERNA2] getDay() ended.')

	#===================
	# Terna spider works
	#===================
	logger.info('[TERNA] getDay() started.')
	for item in TERNA:
		terna = TernaSpider(logger)
		terna.getData(item, date, date)
		terna.driver.quit()
	
	# Logs
	bot('INFO', 'TERNA', 'getDaily ended.')
	logger.info('[TERNA] getDay() ended.')

	#=================
	# GME spider works
	#=================
	logger.info('[GME] getDay() started.')

	gme = GMESpider(logger)
	for item in GME:
		gme.getData(item, date, date)
	for item in GME_NEXT:
		gme.getData(item, date_nxt, date_nxt)
	gme.getData(GME_WEEK, date_week)

	gme.driver.quit()
	
	#===================
	# Kill the processor
	#===================
	processor.stop()
	
	# Logs
	logger.info('[GME] getDay() ended.')
	bot('INFO', 'GME', 'getDaily ended.')

def getHistory():
	"""Visit the GME website and Terna one, call the spiders to retrieve 
	the data from the 01/02/2017 up to the day before the current one and call 
	the file processor to update the file to a MongoDB database.
	"""
	bot('INFO', 'GME', 'getHistory started.')
	processor = FileProcessor(logger)

	start = START
	limit = datetime.now()
	"""
	#=============================================================
	# Terna spider works. From the 01/02/2017 to the current month
	#=============================================================
	logger.info('[TERNA] getHistory() started.')
	while start.strftime('%YY') != limit.strftime('%YY'):
		end = start + relativedelta(years=+1, days=-1)
		for item in TERNA:
			terna = TernaSpider(logger)
			terna.getData(
				item, 
				start.strftime('%d/%m/%Y'), 
				end.strftime('%d/%m/%Y')
			)
			terna.driver.quit()
			del terna
		
		start = end + relativedelta(days=+ 1)
	
	end = datetime.now() + relativedelta(days=-1)
	
	# Terna spider works. From the 1st of the current month to the day before
	# the current one
	for item in TERNA:
		terna = TernaSpider(logger)
		terna.getData(
			item, 
			start.strftime('%d/%m/%Y'), 
			end.strftime('%d/%m/%Y')
		)
		terna.driver.quit()

	# Logs
	logger.info('[TERNA] getHistory() ended.')
	bot('INFO', 'TERNA', 'getHistory ended.')

	#=======================================================================
	# Terna Secondary Reserve spider works. From the starting day to the day
	# before the current one
	#=======================================================================
	logger.info('[TERNA2] getHistory() started.')
	
	terna = TernaReserve()
	terna.getHistory()
	terna.driver.quit()
	
	# Logs 
	bot('INFO', 'TERNA2', 'getHistory ended.')
	logger.info('[TERNA2] getHistory() ended.')
	"""
	#=================
	# GME spider works
	#=================
	gme = GMESpider(logger)
	logger.info('[GME] getHistory() started.')
	start = START
	
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
	if start.strftime('%d/%m/%Y')!=datetime.now().strftime('%d/%m/%Y'):
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
	limit = datetime.now() + relativedelta(days=-8)
	
	while start.strftime('%dd/%mm/%YY') != limit.strftime('%dd/%mm/%YY'):
		gme.getData(GME_WEEK, start.strftime('%d/%m/%Y'))
		start += relativedelta(days=+1)

	gme.driver.quit()
	
	# Logs
	logger.info('[GME] getHistory() ended.')
	processor.stop()
	bot('INFO', 'GME', 'getHistory ended.')