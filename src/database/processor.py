from sys import dont_write_bytecode
from threading import Thread, Event
from pathlib import Path
from src.common.config import DOWNLOAD, DB_NAME, MONGO_STRING, QUEUE
from src.loggerbot.bot import bot
from src.database.csvParse import ParseCsv
from src.database.xmlprocessors import process_file, process_transit_file, process_OffPub
from time import sleep
import motor.motor_asyncio

dont_write_bytecode = True


class FileProcessor(Thread):
    """A thread class used to process .xml  and .xlsx files and send data to 
    the database.

    Parameters
    ----------
    log : logging.logger
        logger instance to display and save logs

    Attributes
    ----------
    log : logging.logger
        logger instance to display and save logs
    db : motor_asyncio.AsyncIOMotorDatabase
        the database to use

    Methods
    -------
    databaseInit()
    run()
    toDatabase(fname)
    sendData(parsed_data, collection)
    """

    def __init__(self, log):
        Thread.__init__(self)
        self.log = log
        self.db = self.databaseInit()
        self.stop_event = Event()
        self.start()

    def databaseInit(self):
        """Initialize the connection to the database.

        Returns
        -------
        db : motor_asyncio.AsyncIOMotorDatabase
            the database to use
        
        """

        try:
            self.log.info("[PROCESS] Attempting to connect to the database...")
            client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_STRING)
            db = client[DB_NAME]
            self.log.info("[PROCESS] Connected to the database.")
            return db
        except Exception as e:
            self.log.error(
                f"[PROCESS] Exception while connecting to the db: {e}"
            )
            # Bot Notification
            bot('ERROR', 'PROCESS', 'Connection failed.')

    def run(self):
        """Method called when the thread starts.
        It runs until the files in the download folder have all been
        processed and sent to the database.
        """

        self.log.info("[PROCESS] Processor Running")
        
        while not self.stop_event.is_set() or not QUEUE.empty():
            fname = QUEUE.get()
            # Processing
            self.toDatabase(fname)
            # Clean folder
            Path(DOWNLOAD + '/' + fname).unlink()
            sleep(.5)

        self.log.info("[PROCESS] Processing Done")
        bot('INFO', 'PROCESSOR', 'Processing Done.')

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

        self.log.info(f"[PROCESS] Processing {fname}")

        if fname[11:-4] == 'OffertePubbliche':
            collection = self.db['OffertePubbliche']
        elif fname[8:11] == 'MGP':
            collection = self.db['MGP']
        elif fname[8:10] == 'MI':
            collection = self.db['MI']
        elif fname[8:11] == 'MSD' or fname[8:11] == 'MBP':
            collection = self.db['MSD']
        elif 'xls' in fname:
            collection = self.db['Terna']

        if fname[11:-4] == 'LimitiTransito' or fname[11:-4] == 'Transiti':
            parsed_data = process_transit_file(fname)
            self.sendData(parsed_data, collection)
        elif fname[11:-4] == 'OffertePubbliche':
            parsed_data = process_OffPub(fname)
            collection.insert_many(parsed_data)
        elif 'xls' in fname:
            df = ParseCsv.excel_to_dic(f"{DOWNLOAD}/{fname}")
            if 'EnergyBal' in fname:
                parsed_data = ParseCsv.to_list_dict(df, 'EnBal')
            elif 'TotalLoad' in fname:
                parsed_data = ParseCsv.to_list_dict(df, 'ToLo')
            elif 'MarketLoad' in fname:
                parsed_data = ParseCsv.to_list_dict(df, 'MaLo')
            else:
                parsed_data = ParseCsv.to_list_dict(df, 'RiSe')
            if parsed_data:
                self.sendData(parsed_data, collection)
        else:
            parsed_data = process_file(fname)
            self.sendData(parsed_data, collection)
        
    def sendData(self, parsed_data, collection):
        """Updates the selected collection with the documents made of paresd
        data.
        
        Parameters
        ----------
        parsed_data : list
            dict list with data to store in the database
        collection : motor_asyncio.AsyncIOMotorCollection
            collection to update
        """
        for item in parsed_data:
            try:
                collection.update_one({'Data':item['Data'], 'Ora':item['Ora']},
                                    {"$set": item},
                                    upsert=True)
            except Exception as e:
                self.log.error(
                    f"[PROCESS] Exception while updating the db: {e}"
                )
                # Bot Notification
                bot('ERROR', 'PROCESSOR', 'Update failed.')
