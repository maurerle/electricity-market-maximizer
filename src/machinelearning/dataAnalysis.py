from sys import dont_write_bytecode
import logging
import logging.config
from src.common.config import MONGO_STRING
from pymongo import MongoClient

dont_write_bytecode = True

class DataAnalysis():
    def __init__ (self, log):
        self.log = log
        try:
            #client = MongoClient(MONGO_STRING)
            client = MongoClient('localhost', 27017)
            self.db = client['InterProj']
            
        except Exception as e:
            self.log.info("Exception while connecting to the db: " + str(e))
    




    
