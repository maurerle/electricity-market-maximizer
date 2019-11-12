import sys
import threading
import logging
import logging.config
from src.common.config import *
from src.loggerbot.bot import bot
from src.database.xmlprocessors import *
import time
from pymongo import MongoClient


sys.dont_write_bytecode = True

class FileProcessor(threading.Thread):
    """A thread class used to process .xml files and send data to the database.
	
    Parameters
    ----------
    log : logging.logger
        logger instance to display and save logs
    target : str
        the type of files to process ('history' or 'daily')

	Attributes
	----------
    target : str
        the type of files to process ('history' or 'daily')
    log : logging.logger
        logger instance to display and save logs
    db : pymongo.database.Database
        the database to use
	
	Methods
	-------
    databaseInit()
    run()
    toDatabase(fname)
	"""

    def __init__(self, log, target):
        threading.Thread.__init__(self)
        self.target = target
        self.log = log
        self.db = self.databaseInit()
        self.start()

    def databaseInit(self):
        """Initialize the connection to the database.

        Returns
		-------
        db : pymongo.database.Database
            the database to use
        """

        try:
            self.log.info("Attempting to connect to the database...")
            client = MongoClient(MONGO_STRING)
            db = client[DB_NAME]
            self.log.info("Connected to the database.")
            return db
        except Exception as e:
            self.log.error("Exception while connecting to the db: " + str(e))
            # Bot Notification
            bot('ERROR', 'GME_MongoClient', 'Connection failed.')
        

    def run(self):
        """Method called when the thread start.
        It runs until the files in the download folder have all been
        processed and sent to the database.
        """

        self.log.info("GME Processor Running")
        file_cnt = 0
        if self.target == 'history':
            LIMIT = H_FILES
        elif self.target == 'daily':
            LIMIT = D_FILES

        while file_cnt < LIMIT:
            # File history managing
            flist = os.listdir(DOWNLOAD)
            flist = list(set(flist) - set([x for x in flist if 'zip' in x or 'void' in x]))
            try:
                fname = flist[0]
                # Processing
                self.toDatabase(fname)
                # Clean folder
                os.remove(DOWNLOAD + '/' + fname)
                file_cnt += 1
            except:
                time.sleep(2)
            time.sleep(.1)
        self.log.info("GME Processing Done")

    def toDatabase(self, fname):
        """Process and send the data to the database.

		Parameters
		----------
        fname : str
            name of the .xml file to process
		"""

        self.log.info(f"Processing {fname}")

        if fname[8:11] == 'MGP':
            collection = self.db['MGP']
        elif fname[8:10] == 'MI':
            collection = self.db['MI']
        elif fname[8:11] == 'MSD' or fname[8:11] == 'MBP':
            collection = self.db['MSD']
        
        if fname[11:-4] == 'LimitiTransito' or fname[11:-4] == 'Transiti':
            data = process_transit_file(fname)
        else:
            data = process_file(fname)

        for item in data.values():
            try:
                collection.update_one({'Data':item['Data'], 'Ora':item['Ora']}, 
                                    {"$set": item}, 
                                    upsert=True)
            except Exception as e:
                self.log.error("Exception while updating the db: " + str(e))
                # Bot Notification
                bot('ERROR', 'GME_MongoClient', 'Update failed.')
