from sys import dont_write_bytecode
import logging
import logging.config
from src.machinelearning.dataAnalysis import DataAnalysis
from src.common.config import MONGO_STRING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pprint
import scipy.signal as sig
from datetime import datetime
import time
import mpl_finance as fin 

dont_write_bytecode = True

logging.config.fileConfig('src/common/logging.conf')
logger = logging.getLogger(__name__)

data = DataAnalysis(logger)

offers = data.db['OffertePubbliche2']

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

def awdOff(): 
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
                        'AWD_PRICE':'$AWARDED_PRICE_NO',
                        'OFF_PRICE':'$ENERGY_PRICE_NO',
                        'TIME':'$Timestamp_Flow'
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
                },{
                    '$sort':{
                        'TIME':1
                    }
                }
            ]
    
    return pipeline

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

def caseStudyOperator(op):
    pipeline = [
                {
                    '$match':{
                        'MARKET_CD':'MGP',
                        'OPERATORE':op,
                        'STATUS_CD':'ACC',
                        'ZONE_CD':'NORD',
                        'ENERGY_PRICE_NO':{
                            '$ne':0
                        },
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

def caseStudyOperatorQ(op):
    pipeline = [
                {
                    '$match':{
                        'MARKET_CD':'MGP',
                        'OPERATORE':op,
                        'STATUS_CD':'ACC',
                        'ZONE_CD':'NORD',
                        'QUANTITY_NO':{
                            '$ne':0
                        },
                        'Timestamp_Flow':{
                            '$gt':0
                        }
                    }
                },{
                    '$project':{
                        '_id':0,
                        'STATUS_CD':1,
                        'OFF_PRICE':'$QUANTITY_NO',
                        'TIME':'$Timestamp_Flow'
                    }
                },{
                    '$sort':{
                        'TIME':1
                    }
                }
            ]
    
    return pipeline

ops = [
    'IREN ENERGIA SPA',
    'ENI SPA',
    'ENEL PRODUZIONE S.P.A.'
]

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
"""
# All the companies, awarded price per zone
fig = plt.figure()
for item in zones:
    ls = []
    temp = offers.aggregate(awdZone(item), allowDiskUse=True)
    print('Aggregation')
    for item2 in temp:
        ls.append(item2)
    x = np.asarray([i['TIME'] for i in ls])
    y = np.asarray([i['AWD_PRICE'] for i in ls])


    plt.plot(x,y, linewidth=.6, label=item)

plt.ylabel('Awarded Price [\u20ac/MWh]')
plt.xlabel('Timestamp')

lgnd = plt.legend(loc="upper left")

for line in lgnd.get_lines():
    line.set_linewidth(2)

plt.show()
"""


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


#====================================================
# All the companies, awarded price per zone. No isles
#====================================================
fig = plt.figure()
#for item in ['NORD', 'CNOR', 'SUD', 'CSUD']:
for item in ['NORD']:
    print(f'Processing zone: {item}')
    cur = offers.aggregate(awdZone(item), allowDiskUse=True)
    df = aggResamp(cur, '12H', 'AWD_PRICE')
        
    plt.plot(
        df.index,
        df['AWD_PRICE']['mean'],
        linewidth=.6, 
        label=item
    )

plt.ylabel('Awarded Price [\u20ac/MWh]')
plt.xlabel('Date')

lgnd = plt.legend(loc="upper left")

for line in lgnd.get_lines():
    line.set_linewidth(2)

plt.show()
exit()
"""

#===============================
# Case of study, three companies 
#===============================
fig = plt.figure()

for item in ops:
    cur = offers.aggregate(caseStudyOperator(item), allowDiskUse=True)
    df = aggResamp(cur, '12H', 'OFF_PRICE')
    plt.plot(
        df.index,
        df['AWD_PRICE']['mean'],
        linewidth=.6, 
        label=item
    )

plt.ylabel('Awarded Price [\u20ac/MWh]')
plt.xlabel('Date')

lgnd = plt.legend(loc="upper left")

for line in lgnd.get_lines():
    line.set_linewidth(2)

plt.show()
exit()
"""   
    plt.errorbar(
        resamp.index,
        resamp['OFF']['mean'], 
        yerr=resamp['OFF']['std'], 
        elinewidth=0.5,# width of error bar line
        ecolor='k',    # color of error bar
        capsize=3,     # cap length for error bar
        capthick=0.7   # cap thickness for error bar
    )
    plt.xticks(rotation='vertical')


for item in ops:
    off = []
    time = []
    temp = offers.aggregate(caseStudyOperatorQ(item), allowDiskUse=True)

    for item2 in temp:
        off.append(item2['OFF_PRICE'])
        time.append(datetime.fromtimestamp(item2['TIME']))
    df = pd.DataFrame({
        'OFF':off,
        'TIME':time
    })
    df = df.set_index(pd.DatetimeIndex(df['TIME']))
    

    resamp = (
    df
    .resample('1H')
    .agg(['std','mean'])
    )
    
    plt.errorbar(
        resamp.index,
        resamp['OFF']['mean'], 
        yerr=resamp['OFF']['std'], 
        elinewidth=0.5,# width of error bar line
        ecolor='k',    # color of error bar
        capsize=3,     # cap length for error bar
        capthick=0.7   # cap thickness for error bar
    )

lgnd = plt.legend(loc="upper left")
plt.ylabel('Average Offered Price [\u20ac/MWh]')
plt.xlabel('Data')





# Difference between the offered price and the awarded price for one company.
fig = plt.figure()
temp = list(offers.aggregate(awdOff()))
x1 = np.asarray([i['OFF_PRICE'] for i in temp if i['OFF_PRICE']<250])
y1 = np.asarray([i['AWD_PRICE'] for i in temp if i['OFF_PRICE']<250])

plt.scatter(x1,y1, linewidth=.6, s=.2, label='offeredV.S.awarded')
plt.plot(x1,x1, linewidth=.6, color='red', label='offered=awarded')

plt.ylabel('Awarded Price [\u20ac/MWh]')
plt.xlabel('Offered Price [\u20ac/MWh]')

lgnd = plt.legend(loc="upper left")

lgnd.get_lines()[0].set_linewidth(1)
lgnd.legendHandles[1]._sizes = [5]
plt.show()

# Rejected of Accepted offers wrt time
fig = plt.figure()
temp = list(offers.aggregate(offStatus(),allowDiskUse=True))
x1 = np.asarray([i['TIME'] for i in temp if i['STATUS_CD']=='ACC' and i['OFF_PRICE']<600])
y1 = np.asarray([i['OFF_PRICE'] for i in temp if i['STATUS_CD']=='ACC'and i['OFF_PRICE']<600])
x2 = np.asarray([i['TIME'] for i in temp if i['STATUS_CD']=='REJ'and i['OFF_PRICE']<600])
y2 = np.asarray([i['OFF_PRICE'] for i in temp if i['STATUS_CD']=='REJ'and i['OFF_PRICE']<600])

plt.scatter(x1,y1, linewidth=.6, s=.3, label='Accepted')
plt.scatter(x2,y2, linewidth=.6, s=.3, label='Rejected')

plt.xlabel('Timestamp')
plt.ylabel('Offered Price [\u20ac/MWh]')
lgnd = plt.legend(loc="upper left")

lgnd.legendHandles[0]._sizes = [5]
lgnd.legendHandles[1]._sizes = [5]

plt.show()

# Rejected of Accepted offers wrt QUANTITY
fig = plt.figure()
temp = list(offers.aggregate(priceQuant()))
y1 = np.asarray([i['QNTY'] for i in temp if i['STATUS_CD']=='ACC' and i['OFF_PRICE']<201 and i['QNTY']<800])
x1 = np.asarray([i['OFF_PRICE'] for i in temp if i['STATUS_CD']=='ACC'and i['OFF_PRICE']<201 and i['QNTY']<800])
y2 = np.asarray([i['QNTY'] for i in temp if i['STATUS_CD']=='REJ'and i['OFF_PRICE']<201 and i['QNTY']<800])
x2 = np.asarray([i['OFF_PRICE'] for i in temp if i['STATUS_CD']=='REJ'and i['OFF_PRICE']<201 and i['QNTY']<800])

plt.scatter(x1,y1, linewidth=.6, s=.4, label='Accepted', alpha=.5)
plt.scatter(x2,y2, linewidth=.6, s=.4, label='Rejected', alpha=.5)

plt.ylabel('Quantity [MWh]')
plt.xlabel('Offered Price [\u20ac/MWh]')
lgnd = plt.legend(loc="upper left")

lgnd.legendHandles[0]._sizes = [5]
lgnd.legendHandles[1]._sizes = [5]

plt.show()
"""

pd.set_option('display.max_rows', 700)
# 3 Companies, offered price per zone
fig = plt.figure()

for item in ops:
    off = []
    time = []
    temp = offers.aggregate(caseStudyOperator(item), allowDiskUse=True)

    for item2 in temp:
        off.append(item2['OFF_PRICE'])
        time.append(datetime.fromtimestamp(item2['TIME']))
    df = pd.DataFrame({
        'OFF':off,
        'TIME':time
    })
    df = df.set_index(pd.DatetimeIndex(df['TIME']))
    

    resamp = (
    df
    .resample('1H')
    .agg(['std','mean'])
    )
    
    plt.errorbar(
        resamp.index,
        resamp['OFF']['mean'], 
        yerr=resamp['OFF']['std'], 
        elinewidth=0.5,# width of error bar line
        ecolor='k',    # color of error bar
        capsize=3,     # cap length for error bar
        capthick=0.7   # cap thickness for error bar
    )
    plt.xticks(rotation='vertical')


for item in ops:
    off = []
    time = []
    temp = offers.aggregate(caseStudyOperatorQ(item), allowDiskUse=True)

    for item2 in temp:
        off.append(item2['OFF_PRICE'])
        time.append(datetime.fromtimestamp(item2['TIME']))
    df = pd.DataFrame({
        'OFF':off,
        'TIME':time
    })
    df = df.set_index(pd.DatetimeIndex(df['TIME']))
    

    resamp = (
    df
    .resample('1H')
    .agg(['std','mean'])
    )
    
    plt.errorbar(
        resamp.index,
        resamp['OFF']['mean'], 
        yerr=resamp['OFF']['std'], 
        elinewidth=0.5,# width of error bar line
        ecolor='k',    # color of error bar
        capsize=3,     # cap length for error bar
        capthick=0.7   # cap thickness for error bar
    )

lgnd = plt.legend(loc="upper left")
plt.ylabel('Average Offered Price [\u20ac/MWh]')
plt.xlabel('Data')



plt.show()

exit()
"""