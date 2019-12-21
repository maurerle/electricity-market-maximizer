from sys import dont_write_bytecode
from src.common.stats import Statistics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams



dont_write_bytecode = True
rcParams.update({'figure.autolayout': True})

data = Statistics()

offers = data.db['OffertePubbliche']


#====================================================
# All the companies, awarded price per zone. No isles
#====================================================
fig = plt.figure()
for item in ['NORD', 'CNOR', 'SUD', 'CSUD']:
    print(f'Processing zone: {item}')
    cur = offers.aggregate(data.awdZone(item, 'AWARDED_PRICE_NO'), allowDiskUse=True)
    df = data.aggResamp(cur, '12H', 'AWD')
        
    plt.plot(
        df.index,
        df['AWD']['mean'],
        linewidth=.6, 
        label=item
    )

plt.ylabel('Awarded Price [\u20ac/MWh]')
plt.xlabel('Date')
plt.grid(linestyle='--', linewidth=.4, which="both")

plt.xticks(rotation='vertical')

lgnd = plt.legend(loc="upper center",ncol = 2)

for line in lgnd.get_lines():
    line.set_linewidth(2)

plt.show()


#=====================================================
# All the companies, awarded price per zone. All zones
#=====================================================
fig = plt.figure()
for item in ['NORD', 'CNOR', 'SUD', 'CSUD', 'SICI', 'SARD']:
    print(f'Processing zone: {item}')
    cur = offers.aggregate(data.awdZone(item, 'AWARDED_PRICE_NO'), allowDiskUse=True)
    df = data.aggResamp(cur, '12H', 'AWD')
        
    plt.plot(
        df.index,
        df['AWD']['mean'],
        linewidth=.6, 
        label=item
    )

plt.ylabel('Awarded Price [\u20ac/MWh]')
plt.xlabel('Date')
plt.grid(linestyle='--', linewidth=.4, which="both")

plt.xticks(rotation='vertical')

lgnd = plt.legend(loc="upper center",ncol = 2)

for line in lgnd.get_lines():
    line.set_linewidth(2)

plt.show()


#====================================================
# Case of study, three companies. Offered Price. NORD 
#====================================================
fig = plt.figure()

for item in ['IREN ENERGIA SPA', 'ENI SPA', 'ENEL PRODUZIONE S.P.A.']:
    cur = offers.aggregate(data.caseStudyOperator(item, 'ENERGY_PRICE_NO'), allowDiskUse=True)
    df = data.aggResamp(cur, '12H', 'OFF')
    plt.plot(
        df.index,
        df['OFF']['mean'],
        linewidth=.6, 
        label=item
    )

plt.xticks(rotation='vertical')
plt.ylabel('Offered Price [\u20ac/MWh]')
plt.xlabel('Date')

lgnd = plt.legend(loc="upper center")

for line in lgnd.get_lines():
    line.set_linewidth(2)

plt.show()

   
#===========================================
# Case study, IREN. Offered Price. All zones 
#===========================================
fig = plt.figure()

cur = offers.aggregate(
    data.caseStudyOperator('IREN ENERGIA SPA', 'ENERGY_PRICE_NO'), 
    allowDiskUse=True
)
df = data.aggResamp(cur, '12H', 'OFF')
plt.errorbar(
    df.index,
    df['OFF']['mean'],
    yerr=df['OFF']['std'],
    linewidth=.6, 
    label='IREN ENERGIA SPA', 
    ecolor='k',  
    capsize=3,  
    capthick=0.7
)

plt.xticks(rotation='vertical')
plt.ylabel('Offered Price [\u20ac/MWh]')
plt.xlabel('Date')

lgnd = plt.legend(loc="upper center")

for line in lgnd.get_lines():
    line.set_linewidth(2)

plt.show()


#=====================================================
# Case study, IREN. Offered Price. All zones separated
#=====================================================
fig = plt.figure()

for item in ['NORD', 'CNOR', 'SUD', 'CSUD', 'SICI', 'SARD']:
    print(f'Processing {item}')
    cur = offers.aggregate(
        data.caseStudyOperatorZone('IREN ENERGIA SPA', item, 'ENERGY_PRICE_NO'), 
        allowDiskUse=True
    )
    df = data.aggResamp(cur, '12H', 'OFF')

    plt.plot(
        df.index,
        df['OFF']['mean'],
        linewidth=.6, 
        label=f'IREN {item}'
    )

plt.xticks(rotation='vertical')
plt.ylabel('Offered Price [\u20ac/MWh]')
plt.xlabel('Date')

lgnd = plt.legend(loc="upper right")

for line in lgnd.get_lines():
    line.set_linewidth(2)

plt.show()

#============================================================
# Difference between the offered price and the awarded price.
#============================================================
fig = plt.figure()
temp = list(offers.aggregate(data.awdOff('AWARDED_PRICE_NO', 'ENERGY_PRICE_NO')))
x1 = np.asarray([i['OFF'] for i in temp if i['OFF']<250])
y1 = np.asarray([i['AWD'] for i in temp if i['OFF']<250])

plt.scatter(x1,y1, linewidth=.6, s=.2, label='offeredV.S.awarded')
plt.plot(x1,x1, linewidth=.6, color='red', label='offered=awarded')

plt.ylabel('Awarded Price [\u20ac/MWh]')
plt.xlabel('Offered Price [\u20ac/MWh]')

lgnd = plt.legend(loc="upper left")
plt.savefig('1.png')
lgnd.get_lines()[0].set_linewidth(1)
lgnd.legendHandles[1]._sizes = [5]

plt.show()

#=====================================
# Rejected of Accepted offers wrt time
#=====================================
fig = plt.figure()
temp = list(offers.aggregate(data.offStatus('ENERGY_PRICE_NO'),allowDiskUse=True))
x1 = np.asarray([i['TIME'] for i in temp if i['STATUS_CD']=='ACC' and i['OFF']<600])
y1 = np.asarray([i['OFF'] for i in temp if i['STATUS_CD']=='ACC'and i['OFF']<600])
x2 = np.asarray([i['TIME'] for i in temp if i['STATUS_CD']=='REJ'and i['OFF']<600])
y2 = np.asarray([i['OFF'] for i in temp if i['STATUS_CD']=='REJ'and i['OFF']<600])

plt.scatter(x1,y1, linewidth=.6, s=.3, label='Accepted')
plt.scatter(x2,y2, linewidth=.6, s=.3, label='Rejected',alpha=.4)

plt.xlabel('Timestamp')
plt.ylabel('Offered Price [\u20ac/MWh]')
lgnd = plt.legend(loc="upper left")

lgnd.legendHandles[0]._sizes = [5]
lgnd.legendHandles[1]._sizes = [5]

plt.show()

#=================================================
# Rejected of Accepted offers wrt offered quantity
#=================================================
fig = plt.figure()
temp = list(offers.aggregate(data.priceQuant()))
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


#=======================================================
# All the companies, awarded quantity per zone. No isles
#=======================================================
fig = plt.figure()
for item in ['NORD', 'CNOR', 'SUD', 'CSUD']:
    print(f'Processing zone: {item}')
    cur = offers.aggregate(data.awdZone(item, 'AWARDED_QUANTITY_NO'), allowDiskUse=True)
    df = data.aggResamp(cur, '12H', 'AWD')
        
    plt.plot(
        df.index,
        df['AWD']['mean'],
        linewidth=.6, 
        label=item
    )

plt.ylabel('Awarded Quantity [MWh]')
plt.xlabel('Date')
plt.grid(linestyle='--', linewidth=.4, which="both")

plt.xticks(rotation='vertical')

lgnd = plt.legend(loc="upper center",ncol = 2)

for line in lgnd.get_lines():
    line.set_linewidth(2)

plt.show()


#========================================================
# All the companies, awarded quantity per zone. All zones
#========================================================
fig = plt.figure()
for item in ['NORD', 'CNOR', 'SUD', 'CSUD', 'SICI', 'SARD']:
    print(f'Processing zone: {item}')
    cur = offers.aggregate(data.awdZone(item, 'AWARDED_QUANTITY_NO'), allowDiskUse=True)
    df = data.aggResamp(cur, '12H', 'AWD')
        
    plt.plot(
        df.index,
        df['AWD']['mean'],
        linewidth=.6, 
        label=item
    )

plt.ylabel('Awarded Quantity [MWh]')
plt.xlabel('Date')
plt.grid(linestyle='--', linewidth=.4, which="both")

plt.xticks(rotation='vertical')

lgnd = plt.legend(loc="upper center",ncol = 2)

for line in lgnd.get_lines():
    line.set_linewidth(2)

plt.show()


#=======================================================
# Case of study, three companies. Offered Quantity. NORD 
#=======================================================
fig = plt.figure()

for item in ['IREN ENERGIA SPA', 'ENI SPA', 'ENEL PRODUZIONE S.P.A.']:
    cur = offers.aggregate(data.caseStudyOperator(item, 'QUANTITY_NO'), allowDiskUse=True)
    df = data.aggResamp(cur, '12H', 'OFF')
    plt.plot(
        df.index,
        df['OFF']['mean'],
        linewidth=.6, 
        label=item
    )

plt.xticks(rotation='vertical')
plt.ylabel('Offered Quantity [MWh]')
plt.xlabel('Date')

lgnd = plt.legend(loc="upper center")

for line in lgnd.get_lines():
    line.set_linewidth(2)

plt.show()

   
#==============================================
# Case study, IREN. Offered Quantity. All zones 
#==============================================
fig = plt.figure()

cur = offers.aggregate(
    data.caseStudyOperator('IREN ENERGIA SPA', 'QUANTITY_NO'), 
    allowDiskUse=True
)
df = data.aggResamp(cur, '12H', 'OFF')
plt.errorbar(
    df.index,
    df['OFF']['mean'],
    yerr=df['OFF']['std'],
    linewidth=.6, 
    label='IREN ENERGIA SPA', 
    ecolor='k',  
    capsize=3,  
    capthick=0.7
)

plt.xticks(rotation='vertical')
plt.ylabel('Offered Quantity [MWh]')
plt.xlabel('Date')

lgnd = plt.legend(loc="upper center")

for line in lgnd.get_lines():
    line.set_linewidth(2)

plt.show()


#========================================================
# Case study, IREN. Offered Quantity. All zones separated
#========================================================
fig = plt.figure()

for item in ['NORD', 'CNOR', 'SUD', 'CSUD', 'SICI', 'SARD']:
    print(f'Processing {item}')
    cur = offers.aggregate(
        data.caseStudyOperatorZone('IREN ENERGIA SPA', item, 'QUANTITY_NO'), 
        allowDiskUse=True
    )
    df = data.aggResamp(cur, '12H', 'OFF')
    print(df)
    plt.plot(
        df.index,
        df['OFF']['mean'],
        linewidth=.6, 
        label=f'IREN {item}'
    )

plt.xticks(rotation='vertical')
plt.ylabel('Offered Quantity [MWh]')
plt.xlabel('Date')

lgnd = plt.legend(loc="upper right")

for line in lgnd.get_lines():
    line.set_linewidth(2)

plt.show()


#=================================================================
# Difference between the offered quantity and the awarded quantity
#=================================================================
fig = plt.figure()
temp = list(offers.aggregate(data.awdOff('AWARDED_QUANTITY_NO', 'QUANTITY_NO')))
x1 = np.asarray([i['OFF'] for i in temp if i['OFF']<250])
y1 = np.asarray([i['AWD'] for i in temp if i['OFF']<250])

plt.scatter(x1,y1, linewidth=.6, s=.2, label='offeredV.S.awarded')
plt.plot(x1,x1, linewidth=.6, color='red', label='offered=awarded')

plt.ylabel('Awarded Quantity [MWh]')
plt.xlabel('Offered Quantity [MWh]')

lgnd = plt.legend(loc="upper left")

lgnd.get_lines()[0].set_linewidth(1)
lgnd.legendHandles[1]._sizes = [5]

plt.show()