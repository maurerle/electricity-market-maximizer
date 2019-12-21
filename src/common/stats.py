from sys import dont_write_bytecode
from pymongo import MongoClient
import pandas as pd
from datetime import datetime

dont_write_bytecode = True

class Statistics():
    def __init__ (self):
        try:
            #client = MongoClient(MONGO_STRING)
            client = MongoClient('smartgridspolito.ddns.net', 27888)
            self.db = client['InterProj']
            
        except Exception as e:
            print("Exception while connecting to the db: " + str(e))
    
    @staticmethod
    def awdZone(zone, AWD):
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
                            'AWD':f"${AWD}",
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
    def awdOff(AWD, OFF): 
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
                            'AWD':f"${AWD}",
                            'OFF':f"${OFF}",
                            'TIME':'$Timestamp_Flow'
                        }
                    }
                ]
        
        return pipeline

    @staticmethod
    def offStatus(OFF): 
        pipeline = [
                    {
                        '$match':{
                            'MARKET_CD':'MGP',
                            'Timestamp_Flow':{
                                '$gt':0
                            }
                        }
                    },{
                        '$project':{
                            '_id':0,
                            'STATUS_CD':1,
                            'OFF':f"${OFF}",
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
    def caseStudyOperator(op, OFF):
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
                            'OFF':f"${OFF}",
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
    def caseStudyOperatorZone(op, zone, OFF):
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
                            'OFF':f'${OFF}',
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
    def aggResamp(cursor, s_freq, *field):
        ls_1 = []
        ls_2 = []
        ls_3 = []

        for item in cursor:
            ls_1.append(datetime.fromtimestamp(item['TIME']))
            ls_2.append(item[field[0]])
            if len(field) == 2:
                ls_3.append(item[field[1]])
        print('List created')
            
        if len(field) == 2:
            df = pd.DataFrame({
                'TIME':ls_1,
                field[0]:ls_2,
                field[1]:ls_3
            })
        elif len(field) == 1:
            df = pd.DataFrame({
                'TIME':ls_1,
                field[0]:ls_2
            })

        df = df.set_index(pd.DatetimeIndex(df['TIME']))
        print('Dataframe created')
        
        resamp = (
            df
            .resample(s_freq)
            .agg(['std','mean']))
        print('Resampled')

        return resamp