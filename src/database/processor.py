import sys
import threading
import logging
import logging.config
from src.common.config import *
from src.database.xmlprocessors import  *
import time
from pymongo import MongoClient


sys.dont_write_bytecode = True

class FileProcessor(threading.Thread):
    def __init__(self, log, target):
        threading.Thread.__init__(self)
        self.target = target
        self.log = log
        self.db = self.databaseInit()
        self.start()

    def databaseInit(self):
        client = MongoClient(MONGO_STRING)
        db = client[DB_NAME]
        
        return db

    def run(self):
        self.log.info("Processor Running")
        file_cnt = 0
        if self.target == 'history':
            LIMIT = H_FILES
        elif self.target == 'daily':
            LIMIT = D_FILES

        while file_cnt<LIMIT:
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

    def toDatabase(self, fname):
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
            collection.update_one({'Data':item['Data'], 'Ora':item['Ora']}, 
                                  {"$set": item}, 
                                  upsert=True)