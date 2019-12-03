from sys import dont_write_bytecode
import logging
import logging.config
from src.machinelearning.dataAnalysis import DataAnalysis
from src.common.config import MONGO_STRING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint

dont_write_bytecode = True

logging.config.fileConfig('src/common/logging.conf')
logger = logging.getLogger(__name__)

data = DataAnalysis(logger)

offers = data.db['OffertePubbliche']
pipeline = [
            {
                '$match':{
                    'OPERATORE':'ENI SPA',
                    '$or':[
                        {'STATUS_CD':'ACC'}
                    ],
                    'MARKET_CD':'MGP',
                    'ZONE_CD':'SUD',
                    #'INTERVAL_NO':1
                }
            },{
                '$project':{
                    '_id':0,
                    'STATUS_CD':1,
                    'PRICE':{
                        '$multiply':[
                            '$AWARDED_PRICE_NO',
                            '$AWARDED_QUANTITY_NO'
                        ]
                    },
                    'AWARDED_PRICE_NO':1,
                    'Timestamp_Flow':1
                }
            },{
                '$group':{
                    '_id':{
                        'time':'$Timestamp_Flow'
                    },
                    'TOT':{
                        '$sum':'$AWARDED_PRICE_NO'
                    }
                }
            },{
                '$sort':{
                    '_id.time':1
                }
            }
        ]

temp = list(offers.aggregate(pipeline))

x = np.asarray([i['_id']['time'] for i in temp])
y = np.asarray([i['TOT'] for i in temp])

fig = plt.figure()
plt.plot(x,y, linewidth=.5)
plt.show()