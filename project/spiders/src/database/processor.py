from sys import dont_write_bytecode
from threading import Thread, Event
from pathlib import Path
from src.common.config import DOWNLOAD, QUEUE
from src.database.xmlprocessors import process_OffPub
from src.database.csvParse import *
from time import sleep
from influxdb import InfluxDBClient
import os
from datetime import datetime
dont_write_bytecode = True


class FileProcessor(Thread):
    """Thread class to process and send data to the database.
    
    Parameters
    ----------
    log : logging.Logger
        Event logger

    Attributes
    ----------
    log : logging.Logger
        Event logger
    db : influxdb.InfluxDBClient
        InfluxDB Client
    stop_event : threading.Event
        Event to stop the processor
    
    Methods
    -------
    databaseInit()
        Database initialization
    run()
        Run the thread
    stop()
        Set the stop event
    toDatabase(fname)
        Choose the correct file processor and sender
    sendTerna(date, load)
        Send Terna data
    sendData(dem, sup)
        Send GME data
    """
    def __init__(self, log):
        Thread.__init__(self)
        self.log = log
        self.db = self.databaseInit()
        self.stop_event = Event()
        self.start()
        self.working = False
        self.server = None
        
    def databaseInit(self):
        """Database initialization.
        
        Returns
        -------
        influxdb.InfluxDBClient
            InfluxDB Client
        """
        try:
            self.log.info("[PROCESS] Attempting to connect to the database...")
            # define ssh tunnel
            clientID = 'PublicBids'

            db = InfluxDBClient('172.28.5.1', 8086, 'root', 'root', clientID)
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
        #bot('INFO', 'PROCESSOR', 'Processing Done.')

    def stop(self):
        """Set the stop event"""
        self.stop_event.set()

    def toDatabase(self, fname):
        """Choose the correct file processor and sender

        Parameters
        ----------
        fname : str
            name of the file to process
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
        elif 'xls' in fname:
            try:
                date, load = ParseCsv(f"{DOWNLOAD}/{fname}")
                self.sendTerna(date, load)
            except Exception as e:
                print(e)
                #print('Skipped')

    def sendTerna(self, date, load):
        """Send Terna data to the database.
        
        Parameters
        ----------
        date : datetime.datetime
            Date
        load : float
            Load
        """
        body = [{
            'measurement':'STRes',
            'time':date,
            'fields':{
                'threshold':load
            }
        }]
        self.db.write_points(body, time_precision='h')


    def sendData(self, dem, sup):
        """Send GME data to the database.
        
        Parameters
        ----------
        dem : pandas.DataFrame
            Dataframe containing the demand data
        sup : pandas.DataFrame
            Dataframe containing the supply data
        """
        for op in dem.index:
            body = [{
                'tags':{
                    'op':op.upper()
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
            try:
                self.db.write_points(body, time_precision='h')
            except Exception as e:
                print(e)
        for op in sup.index:
            body = [{
                'tags':{
                    'op':op.upper()
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
