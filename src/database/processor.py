import sys
import threading
from pathlib import Path
from src.common.config import DOWNLOAD, DB_NAME, MONGO_STRING, QUEUE
from src.loggerbot.bot import bot
from src.database.csvParse import *
from src.database.xmlprocessors import process_file, process_transit_file, process_OffPub
import time
import motor.motor_asyncio

sys.dont_write_bytecode = True


class FileProcessor(threading.Thread):
    """A thread class used to process .xml files and send data to the database.

    Parameters
    ----------
    log : logging.logger
        logger instance to display and save logs

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
    toDatabase(fname)
    """

    def __init__(self, log):
        threading.Thread.__init__(self)
        self.log = log
        self.db = self.databaseInit()
        self.stop_event = threading.Event()
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
            client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_STRING)
            db = client[DB_NAME]
            self.log.info("Connected to the database.")
            return db
        except Exception as e:
            self.log.error("Exception while connecting to the db: " + str(e))
            # Bot Notification
            bot('ERROR', 'GME_MongoClient', 'Connection failed.')

    def run(self):
        """Method called when the thread starts.
        It runs until the files in the download folder have all been
        processed and sent to the database.
        """

        self.log.info("GME Processor Running")
        
        while not self.stop_event.is_set() or not QUEUE.empty():
            fname = QUEUE.get()
            # Processing
            self.toDatabase(fname)
            # Clean folder
            Path(DOWNLOAD + '/' + fname).unlink()
            time.sleep(.5)

        self.log.info("GME Processing Done")

    def stop(self):
        """Set the stop event"""
        self.stop_event.set()

    def toDatabase(self, fname):
        """Process and send the data to the database.

        Parameters
        ----------
        fname : str
            name of the .xml file to process
        """

        self.log.info(f"Processing {fname}")

        if fname[11:-4] == 'OffertePubbliche':
            collection = self.db['OffertePubbliche']
        elif fname[8:11] == 'MGP':
            collection = self.db['MGP']
        elif fname[8:10] == 'MI':
            collection = self.db['MI']
        elif fname[8:11] == 'MSD' or fname[8:11] == 'MBP':
            collection = self.db['MSD']

        if fname[11:-4] == 'LimitiTransito' or fname[11:-4] == 'Transiti':
            parsed_data = process_transit_file(fname)
            self.sendData(parsed_data, collection)
        elif fname[11:-4] == 'OffertePubbliche':
            parsed_data = process_OffPub(fname)
            collection.insert_many(parsed_data)
        elif 'xlsx' in fname:
            pass
        else:
            parsed_data = process_file(fname)
            self.sendData(parsed_data, collection)

    def sendData(self, parsed_data, collection):
        for item in parsed_data:
            try:
                collection.update_one({'Data':item['Data'], 'Ora':item['Ora']},
                                    {"$set": item},
                                    upsert=True)
            except Exception as e:
                self.log.error("Exception while updating the db: " + str(e))
                # Bot Notification
                bot('ERROR', 'GME_MongoClient', 'Update failed.')
