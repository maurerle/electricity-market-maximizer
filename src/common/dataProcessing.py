from sshtunnel import SSHTunnelForwarder
import pymongo
from src.common.config import DB_NAME, MONGO_HOST
from src.loggerbot.bot import bot
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta

class DataProcessing():
    def __init__(self, log, user, passwd):
        self.log = log
        self.user = user
        self.passwd = passwd
        self.server = None 
        self.db = self.mongo()

    def mongo(self):
        try:
            self.log.info("[MONGO] Attempting to connect to the database...")
            # define ssh tunnel
            self.server = SSHTunnelForwarder(
                MONGO_HOST,
                ssh_username=self.user,
                ssh_password=self.passwd,
                remote_bind_address=('127.0.0.1', 27017),
                local_bind_address=('127.0.0.1', 27017)
            )

            # start ssh tunnel
            self.server.start()
            client = pymongo.MongoClient('127.0.0.1', 27017)
            db = client[DB_NAME]

            self.log.info("[MONGO] Connected to the database.")
            
            return db

        except Exception as e:
            self.log.error(
                f"[MONGO] Exception while connecting to the db: {e}"
            )
            # Bot Notification
            bot('ERROR', 'MONGO', 'Connection failed.')
            
    def mongoStop(self):
        self.server.stop()
    
    def operatorAggregate(self, op, collection):
        """
        pipeline = [
            {
                '$match': {
                    'MARKET_CD':'MGP',
                    'OPERATORE':op,
                    'STATUS_CD':'ACC',
                    'Timestamp_Flow':{
                        '$gte':1543615200.0
                    }
                }
            },{
                '$project': {
                    '_id':0,
                    'GAIN':{
                        '$multiply':[
                            '$ENERGY_PRICE_NO',
                            '$QUANTITY_NO'
                        ]
                    },
                    'AWD_GAIN':{
                        '$multiply':[
                            '$AWARDED_PRICE_NO',
                            '$AWARDED_QUANTITY_NO'
                        ]
                    },
                    'QUANTITY_NO':1,
                    'AWARDED_QUANTITY_NO':1,
                    'ENERGY_PRICE_NO':1,
                    'AWARDED_PRICE_NO':1,
                    'BID_OFFER_DATE_DT':1,
                    'ZONE_CD':1,
                    'PURPOSE_CD':1,
                    'INTERVAL_NO':1
                }
            },{
                '$group': {
                    '_id':{
                        'DATE':'$BID_OFFER_DATE_DT',
                        'HOUR':'$INTERVAL_NO',
                        'TYPE':'$PURPOSE_CD'
                    },
                    'DLY_QTY':{
                        '$sum':'$QUANTITY_NO'
                    },
                    'DLY_GAIN':{
                        '$sum':'$GAIN'
                    },
                    'DLY_AWD_QTY':{
                        '$sum':'$AWARDED_QUANTITY_NO'
                    },
                    'DLY_AWD_GAIN':{
                        '$sum':'$AWD_GAIN'
                    }
                }
            },{
                '$addFields': {
                    "TYPE": '$_id.TYPE',
                    "DATE": '$_id.DATE',
                    'HOUR': '$_id.HOUR'
                }
            }
        ]

        cursor = self.db[collection].aggregate(pipeline)
        temp = [x for x in cursor]
        df = pd.DataFrame(temp)
        """
        df = pd.read_csv('tempDate.csv')
        h = df['HOUR'].astype(int).astype(str).copy()
        temp = df["DATE"].astype(str).str.cat(h, sep =":")
        df['DLY_PRICE'] = df['DLY_GAIN']/df['DLY_QTY']
        df['DLY_AWD_PRICE'] = df['DLY_AWD_GAIN']/df['DLY_AWD_QTY']
        df = df.fillna(0.0)
        date_l = []
        for i in temp:
            if(':24') in i:
                i=i.replace(':24', ':00')
                dtime = datetime.strptime(i, '%Y%m%d:%H') + relativedelta(days=1)
            else:
                dtime = datetime.strptime(i, '%Y%m%d:%H')
            date_l.append(dtime)
 
        df = df.set_index(pd.to_datetime(date_l))
        df = df.drop(columns=['_id', 'DLY_GAIN', 'DLY_AWD_GAIN', 'DATE', 'HOUR', 'Unnamed: 0'])

        df = df.loc[df['TYPE'] == 'BID']
        df = df.sort_index()

        bot('INFO','MONGO', 'Aggregation Finished.')
        return df

    def mgpAggregate(self, timestamp):
        pipeline = [
            {
                '$match':{
	                'Timestamp':{
		                '$gte':timestamp
                    }
                }
            }
        ]

        cursor = self.db['MGP'].aggregate(pipeline)
        temp = [x for x in cursor]
        df = pd.DataFrame(temp)

        drop = ['_id', 'Data', 'Ora', 'Timestamp']
        cols = list(df.columns)
        for col in cols:
            if 'PrezzoConv' in col \
                or 'Coefficiente' in col \
                or '_BID' in col \
                or '_OFF' in col:
                    drop.append(col)
        df = df.set_index(pd.to_datetime(df['Timestamp'], unit='s'))
        df = df.drop(columns=drop).fillna(0.0)
        #df = df.resample('D').mean()
        df = df.sort_index()

        return df

    def merge(self, df1, df2):
        
        merged = pd.merge(df1, df2, left_index=True, right_index=True)
        
        return merged
    
    