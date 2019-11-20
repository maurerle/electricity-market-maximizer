import sys
import pathlib
import threading
import logging
import logging.config
from src.common.config import *
from src.loggerbot.bot import bot
from src.database.csvParse import *
import time
import motor.motor_asyncio
import asyncio

sys.dont_write_bytecode = True

class FileProcessorCSV(threading.Thread):
    """A thread class used to process .xlsx files and send data to the database.

    Parameters
    ----------
    log : logging.logger
        logger instance to display and save logs

    collection : str
        the name of the MongoDB collection must be created.

	Attributes
	----------

    log : logging.logger
        logger instance to display and save logs
    db : pymongo.database.Database
        the database to use

	Methods
	-------
    databaseInit()
    run()
    to_database()
    to_insert()
	"""

    def __init__(self, log, collection):
        threading.Thread.__init__(self)
        self.log = log
        self.collection = collection
        self.db = self.database_init()
        self.stop_event = threading.Event()

        self.start()

    def database_init(self):
        """Initialize the connection to the database.

        Returns
		-------
        db : pymongo.database.Database
            the database to use
        """

        try:
            self.log.info("Attempting to connect to the database...")
            client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_STRING)
            db = client[DB_NAME]
            self.log.info('Database obtained.')
            return db
        except Exception as e:
            self.log.error("Exception while connecting to the db: " + str(e))
            # Bot Notification
            bot('ERROR', 'TERNA_MongoClient', 'Connection failed.')

    def run(self):
        """Method called when the thread start.
        It runs until the files in the download folder have all been
        processed and sent to the database.
        """

        self.log.info("Processor Running")

        while not self.stop_event.is_set() or not QUEUE_TERNA.empty():
            fname = QUEUE_TERNA.get()
            # Processing
            self.toDatabase(fname)
            # Clean folder
            p = pathlib.Path(DOWNLOAD + '/' + fname)
            p.unlink()
            time.sleep(.3)

        self.log.info("TERNA Processing Done")



    async def to_insert(self, document, collection):
        result = await self.db.collection.insert_one(document)
        
    def toDatabase(self, fname):
        """it processes and sends documents to the database.
        :param fname = file path
        """

        self.log.info(f"Processing files: {fname}")
        flag = ParseCsv.find_name(fname)
        try:
            df = ParseCsv.excel_to_dic(f"{DOWNLOAD}/{fname}")
            _dict = ParseCsv.to_dict(df, flag)
            self.log.info(f"Parsed file {fname}")
        except Exception as e:
            self.log.error(f"the file {fname} can not be parsed." + str(e))

        asyncio.run(self.to_insert(_dict, self.collection))

        #loop = asyncio.get_event_loop()
        #loop.run_until_complete(self.to_insert(_dict, self.collection))
