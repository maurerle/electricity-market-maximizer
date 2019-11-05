import sys
import threading
from common.config import *
import common.config as conf
import time
import pymongo
import xmltodict

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
        # To be changed with Polito cluster credentials

        self.client = pymongo.MongoClient(
            "mongodb+srv://new-user:politomerda@cluster0-awyti.mongodb.net/test?retryWrites=true&w=majority"
            )
        self.db = client['InterProj']
        self.collectionMGP = self.db['MGP']
        self.collectionMI = self.db['MI']
        self.collectionMSD = self.db['MSD']



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

        collection = client['InterProj'][fname[8:12]]

        with open(DOWNLOAD + '/' + fname, 'r') as file:
            data = file.read()
            dict = xmltodict.parse(data)["NewDataSet"]
            del dict["xs:schema"]
            # jsonString = json.dumps(dict, indent=4)
            collection.insert_one(dict)

        print("[TODO] Processing {}".format(fname))