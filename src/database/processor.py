import sys
import threading
from common.config import *
import common.config as conf
from database.xmlprocessors import  *
import time
from pymongo import MongoClient


sys.dont_write_bytecode = True

class ProcessorTh(threading.Thread):
    def __init__(self, log):
        threading.Thread.__init__(self)
        self.name = 'processor'
        self.log = log
        self.running = False
        self.databaseInit()
        self.start()

    def databaseInit(self):
        self.client = MongoClient(MONGO_STRING)
        self.db = self.client['InterProj']

    def run(self):
        self.log.info("Processor Running")
        while True:
            if len(conf.HISTORY) != 0:
                self.running = True
                # File history managing
                fname = conf.HISTORY.pop(0)
                # Processing
                self.toDatabase(fname)
                # Clean folder
                os.remove(DOWNLOAD + '/' + fname)
            else:
                if self.running:
                    self.running = False
                    self.log.info("Processing done")
            time.sleep(.1)

    def toDatabase(self, fname):

        print("Processing {}".format(fname))

        if fname[8:11] == 'MGP':
            collection = self.db['MGP']
        elif fname[8:10] == 'MI':
            collection = self.db['MI']
        elif fname[8:11] == 'MSD':
            collection = self.db['MSD']
        
        if fname[11:-4] == 'LimitiTransito' or fname[11:-4] == 'Transiti':
            data = process_transit_file(fname)
        else:
            data = process_file(fname)

        for item in data.values():
            collection.update_one({'Data':item['Data'], 'Ora':item['Ora']}, 
                                  {"$set": item}, 
                                  upsert=True)
        