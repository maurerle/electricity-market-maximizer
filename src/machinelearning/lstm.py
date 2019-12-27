import pandas as pd
from src.common.dataProcessing import DataProcessing
import logging
import logging.config

logging.config.fileConfig('src/common/logging.conf')
logger = logging.getLogger(__name__)

class MGP():
    def __init__ (self, user, passwd):
        self.user = user
        self.passwd = passwd

    def createSet(self):
        mongo = DataProcessing(logger, self.user, self.passwd)
        dataset = mongo.merge(
            mongo.mgpAggregate(1543615200.0),
            mongo.operatorAggregate('IREN ENERGIA SPA', 'OffertePubbliche')
        )
        dataset.to_csv('datasetTest.csv')

class MI():
    def __init__ (self):
        pass

class MSD():
    def __init__ (self):
        pass