from sys import dont_write_bytecode
import logging
import logging.config
from src.common.config import MONGO_STRING
from pymongo import MongoClient

dont_write_bytecode = True

class DataAnalysis():
    def __init__ (self):
        try:
            #client = MongoClient(MONGO_STRING)
            client = MongoClient('localhost', 27017)
            self.db = client['InterProj']
            
        except Exception as e:
            print("Exception while connecting to the db: " + str(e))
    
    @staticmethod
    def awdZone(zone):
        pipeline = [
                    {
                        '$match':{
                            'STATUS_CD':'ACC',
                            'MARKET_CD':'MGP',
                            'ZONE_CD':zone,
                            'Timestamp_Flow':{
                                '$gt':0
                            }
                        }
                    },{
                        '$project':{
                            '_id':0,
                            'AWD_PRICE':'$AWARDED_PRICE_NO',
                            'TIME':'$Timestamp_Flow'
                        }
                    },{
                        '$sort':{
                            'TIME':1
                        }
                    }
                ]
        
        return pipeline

    @staticmethod
    def awdOff(awd, off): 
        pipeline = [
                    {
                        '$match':{
                            'STATUS_CD':'ACC',
                            'MARKET_CD':'MGP',
                        }
                    },{
                        '$project':{
                            '_id':0,
                            'STATUS_CD':1,
                            'AWD':f'${awd}',
                            'OFF':f'${off}',
                            'TIME':'$Timestamp_Flow'
                        }
                    }
                ]
        
        return pipeline

    @staticmethod
    def offStatus(): 
        pipeline = [
                    {
                        '$match':{
                            'MARKET_CD':'MGP',
                        }
                    },{
                        '$project':{
                            '_id':0,
                            'STATUS_CD':1,
                            'AWD_PRICE':'$AWARDED_PRICE_NO',
                            'OFF_PRICE':'$ENERGY_PRICE_NO',
                            'TIME':'$Timestamp_Flow'
                        }
                    },{
                        '$sort':{
                            'TIME':1
                        }
                    }
                ]
        
        return pipeline

    @staticmethod
    def priceQuant(): 
        pipeline = [
                    {
                        '$match':{
                            'MARKET_CD':'MGP',
                        }
                    },{
                        '$project':{
                            '_id':0,
                            'STATUS_CD':1,
                            'QNTY':'$QUANTITY_NO',
                            'OFF_PRICE':'$ENERGY_PRICE_NO',
                            'TIME':'$Timestamp_Flow'
                        }
                    }
                ]
        
        return pipeline

    @staticmethod
    def caseStudyOperator(op):
        pipeline = [
                    {
                        '$match':{
                            'MARKET_CD':'MGP',
                            'OPERATORE':op,
                            'STATUS_CD':'ACC',
                            #'ZONE_CD':'NORD',
                            'Timestamp_Flow':{
                                '$gt':0
                            }
                        }
                    },{
                        '$project':{
                            '_id':0,
                            'STATUS_CD':1,
                            'OFF_PRICE':'$ENERGY_PRICE_NO',
                            'TIME':'$Timestamp_Flow'
                        }
                    },{
                        '$sort':{
                            'TIME':1
                        }
                    }
                ]
        
        return pipeline
    
    @staticmethod
    def caseStudyOperatorZone(op, zone):
        pipeline = [
                    {
                        '$match':{
                            'MARKET_CD':'MGP',
                            'OPERATORE':op,
                            'STATUS_CD':'ACC',
                            'ZONE_CD':zone,
                            'Timestamp_Flow':{
                                '$gt':0
                            }
                        }
                    },{
                        '$project':{
                            '_id':0,
                            'STATUS_CD':1,
                            'OFF_PRICE':'$ENERGY_PRICE_NO',
                            'TIME':'$Timestamp_Flow'
                        }
                    },{
                        '$sort':{
                            'TIME':1
                        }
                    }
                ]
        
        return pipeline