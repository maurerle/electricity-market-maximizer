import threading
from common.config import *
import common.config as conf
import time
import pymongo
import json
import xmltodict


class ProcessorTh(threading.Thread):
    def __init__(self, log):
        threading.Thread.__init__(self)
        self.name = 'processor'
        self.log = log
        self.running = False
        self.start()

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

        # To be changed with Polito cluster credentials

        client = pymongo.MongoClient(
            "mongodb+srv://new-user:politomerda@cluster0-awyti.mongodb.net/test?retryWrites=true&w=majority")

        collection = client['InterProj'][fname[8:-4]]

        with open(DOWNLOAD + '/' + fname, 'r') as file:
            data = file.read()
            dict = xmltodict.parse(data)["NewDataSet"]
            del dict["xs:schema"]
            # jsonString = json.dumps(dict, indent=4)
            collection.insert_one(dict)

        print("[TODO] Processing {}".format(fname))