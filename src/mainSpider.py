from sys import dont_write_bytecode
import logging
import logging.config
from src.loggerbot.bot import bot
from src.database.processor import FileProcessor
from dateutil.relativedelta import relativedelta
from datetime import datetime
from src.spiders.terna import TernaReserve
from src.spiders.gme import GMESpider
from src.common.config import GME_WEEK, START
from sys import argv as args
import getpass

dont_write_bytecode = True

logging.config.fileConfig('src/common/logging.conf')
logger = logging.getLogger(__name__)

def getDay():
	"""Visit the GME website and Terna one, call the spiders to retrieve 
	the daily data and call the file processor to update the file to a MongoDB
	database.
	"""
		
	# Classes init
	processor = FileProcessor(logger, user, passwd)
	
	# Date creation
	date_week = (datetime.now() + relativedelta(days=-7)).strftime('%d/%m/%Y')
	
	#=====================================
	# Terna Secondary Reserve spider works
	#=====================================
	#logger.info('[TERNA2] getDay() started.')
	
	terna = TernaReserve()
	terna.getDaily()
	terna.driver.quit()
	
	# Logs 
	#bot('INFO', 'TERNA2', 'getDaily ended.')
	logger.info('[TERNA2] getDay() ended.')
	
	#=================
	# GME spider works
	#=================
	logger.info('[GME] getDay() started.')

	gme = GMESpider(logger)

	gme.getData(GME_WEEK, date_week)

	gme.driver.quit()
	
	#===================
	# Kill the processor
	#===================
	processor.stop()
	
	# Logs
	logger.info('[GME] getDay() ended.')
	
def getHistory():
	"""Visit the GME website and Terna one, call the spiders to retrieve 
	the data from the 01/02/2017 up to the day before the current one and call 
	the file processor to update the file to a MongoDB database.
	"""
	bot('INFO', 'GME', 'getHistory started.')
	processor = FileProcessor(logger, user, passwd)
	
	start = START
	limit = datetime.now()
	
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
	
	#=================
	# GME spider works
	#=================
	logger.info('[GME] getHistory() started.')
	
	# Get the public offers referred to one week before the current
	start = START
	limit = datetime.now() + relativedelta(days=-8)
	
	while start.strftime('%dd/%mm/%YY') != limit.strftime('%dd/%mm/%YY'):
		gme = GMESpider(logger)
		gme.getData(GME_WEEK, start.strftime('%d/%m/%Y'))
		start += relativedelta(days=+1)
		gme.driver.quit()
	
	# Logs
	logger.info('[GME] getHistory() ended.')
	
	processor.stop()
	
	bot('INFO', 'GME', 'getHistory ended.')

global user 
global passwd 

user = input('Insert SSH username:\n')
passwd = getpass.getpass(prompt='Insert SSH passwd:\n')
