from sys import dont_write_bytecode
from threading import Thread, Event
from pathlib import Path
from src.common.config import DOWNLOAD, DB_NAME, QUEUE, MONGO_HOST
from src.loggerbot.bot import bot
from src.database.xmlprocessors import process_OffPub
from time import sleep
from influxdb import InfluxDBClient
import os
from datetime import datetime
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

    def __init__(self, log, user, passwd):
        Thread.__init__(self)
        self.log = log
        self.user = user
        self.passwd = passwd
        self.db = self.databaseInit()
        self.stop_event = Event()
        self.start()
        self.working = False
        self.server = None
        
    def databaseInit(self):
        """Initialize the connection to the database.

        Returns
        -------
        db : 
            the database to use
        
        """

        try:
            self.log.info("[PROCESS] Attempting to connect to the database...")
            # define ssh tunnel
            clientID = 'PublicBids'

            db = InfluxDBClient('localhost', 8086, 'root', 'root', clientID)
            if {'name': clientID} not in db.get_list_database():
                db.create_database(clientID)

            self.log.info("[PROCESS] Connected to the database.")
            
            return db

        except Exception as e:
            self.log.error(
                f"[PROCESS] Exception while connecting to the db: {e}"
            )
            # Bot Notification
            #bot('ERROR', 'PROCESS', 'Connection failed.')

    def run(self):
        """Method called when the thread starts.
        It runs until the files in the download folder have all been
        processed and sent to the database.
        """

        self.log.info("[PROCESS] Processor Running")
        
        while not self.stop_event.is_set() or not QUEUE.empty():
            fname = QUEUE.get()
            try:
                self.toDatabase(fname)
                # Clean folder
                Path(DOWNLOAD + '/' + fname).unlink()
            except ValueError:
                #bot('ERROR', 'PROCESSOR', f'{fname} skipped.')
                print('Value Error')
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
            dem, sup = process_OffPub(fname)
            if not isinstance(dem, int):
                done = False
                while not done:
                    try:
                        self.sendData(dem, sup)
                        done = True
                    except:
                        sleep(1)   

    def sendData(self, dem, sup):
        """Updates the selected collection with the documents made of paresd
        data.
        
        Parameters
        ----------
        parsed_data : list
            dict list with data to store in the database
        collection : motor_asyncio.AsyncIOMotorCollection
            collection to update
        """
        for op in dem.index:
            body = [{
                'tags':{
                    'op':op
                },
                'measurement':f'demand{dem.loc[op].MARKET}',
                'time':datetime.strptime(
                    dem.loc[op].DATE,
                    '%Y%m%d'
                ),
                'fields':{
                    'Price':dem.loc[op].P,
                    'Quantity':dem.loc[op].Q,
                }
            }]
            self.db.write_points(body, time_precision='h')

        for op in sup.index:
            body = [{
                'tags':{
                    'op':op
                },
                'measurement':f'supply{sup.loc[op].MARKET}',
                'time':datetime.strptime(
                    sup.loc[op].DATE,
                    '%Y%m%d'
                ),
                'fields':{
                    'Price':sup.loc[op].P,
                    'Quantity':sup.loc[op].Q,
                }
            }]
            self.db.write_points(body, time_precision='h')