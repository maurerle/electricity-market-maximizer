from sys import dont_write_bytecode
import logging
import logging.config
from src.machinelearning.dataAnalysis import DataAnalysis
from src.common.config import MONGO_STRING
import pandas as pd
import pprint

dont_write_bytecode = True

logging.config.fileConfig('src/common/logging.conf')
logger = logging.getLogger(__name__)

data = DataAnalysis(logger)

collection = data.db['Terna']
pprint.pprint(collection.find_one())
collection = data.db['MGP']
pprint.pprint(collection.find_one())
collection = data.db['MI']
pprint.pprint(collection.find_one())
collection = data.db['MSD']
pprint.pprint(collection.find_one())
collection = data.db['OffertePubbliche']
pprint.pprint(collection.find_one())