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

def awdZone(zone):
    pipeline = [
                {
                    '$match':{
                        '$or':[
                            {'STATUS_CD':'ACC'}
                        ],
                        'MARKET_CD':'MGP',
                        'ZONE_CD':zone,
                    }
                },{
                    '$project':{
                        '_id':0,
                        'STATUS_CD':1,
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

def awdOff(zone): 
    pipeline = [
                {
                    '$match':{
                        '$or':[
                            {'STATUS_CD':'ACC'}
                        ],
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
                }
            ]
    
    return pipeline


zones = [
    'NORD',
    'SUD',
    'CNOR',
    'CSUD',
    'SICI',
    'SARD'
]

#===================
# PRICES
#===================
# All the companies, awarded price per zone
fig = plt.figure()
for item in zones:
    temp = list(offers.aggregate(awdZone(item)))

    x = np.asarray([i['TIME'] for i in temp])
    y = np.asarray([i['AWD_PRICE'] for i in temp])

    plt.plot(x,y, linewidth=.6, label=item)

plt.ylabel('Awarded Price [\u20ac/MWh]')
plt.xlabel('Timestamp')

lgnd = plt.legend(loc="upper left")

for line in lgnd.get_lines():
    line.set_linewidth(2)

plt.show()

# All the companies, awarded price per zone. No SICI and SARD
fig = plt.figure()
for item in zones:
    if item != 'SICI' and item != 'SARD':
        temp = list(offers.aggregate(awdZone(item)))

        x = np.asarray([i['TIME'] for i in temp])
        y = np.asarray([i['AWD_PRICE'] for i in temp])

        plt.plot(x,y, linewidth=.6, label=item)

plt.ylabel('Awarded Price [\u20ac/MWh]')
plt.xlabel('Timestamp')

lgnd = plt.legend(loc="upper left")

for line in lgnd.get_lines():
    line.set_linewidth(2)

plt.show()

# Difference between the offered price and the awarded price for one company.
fig = plt.figure()
temp = list(offers.aggregate(awdOff('ACC')))
x1 = np.asarray([i['OFF_PRICE'] for i in temp if i['OFF_PRICE']<250])
y1 = np.asarray([i['AWD_PRICE'] for i in temp if i['OFF_PRICE']<250])

plt.scatter(x1,y1, linewidth=.6, s=.2, label='offeredV.S.awarded')
plt.plot(x1,x1, linewidth=.6, color='red', label='offered=awarded')

plt.ylabel('Awarded Price [\u20ac/MWh]')
plt.xlabel('Offered Price [\u20ac/MWh]')

plt.gca().legend(
    ('offeredV.S.awarded', 'offered=awarded'),
    loc = 'upper left'
)

plt.show()

# Rejected of Accepted offers wrt time
fig = plt.figure()
temp = list(offers.aggregate(offStatus()))
x1 = np.asarray([i['TIME'] for i in temp if i['STATUS_CD']=='ACC' and i['OFF_PRICE']<600])
y1 = np.asarray([i['OFF_PRICE'] for i in temp if i['STATUS_CD']=='ACC'and i['OFF_PRICE']<600])
x2 = np.asarray([i['TIME'] for i in temp if i['STATUS_CD']=='REJ'and i['OFF_PRICE']<600])
y2 = np.asarray([i['OFF_PRICE'] for i in temp if i['STATUS_CD']=='REJ'and i['OFF_PRICE']<600])

plt.scatter(x1,y1, linewidth=.6, s=.4, label='Accepted')
plt.scatter(x2,y2, linewidth=.6, s=.4, label='Rejected')

plt.xlabel('Hour of Day')
plt.ylabel('Offered Price [\u20ac/MWh]')
lgnd = plt.legend(loc="upper left")

lgnd.legendHandles[0]._sizes = [5]
lgnd.legendHandles[1]._sizes = [5]

plt.show()
