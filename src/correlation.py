from sys import dont_write_bytecode
from src.common.stats import Statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns



dont_write_bytecode = True
rcParams.update({'figure.autolayout': True})

dat = Statistics()

offers = dat.db['OffertePubbliche2']
mgp = dat.db['MGP']

categ = [
    'OPERATORE', 'ZONE_CD', 'STATUS_CD'
]
pipeline1 = [
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
            'AWARDED_QUANTITY_NO':1,
            'ENERGY_PRICE_NO':1,
            'OPERATORE':1,
            'QUANTITY_NO':1,
            'TIME':'$Timestamp_Flow',
            'STATUS_CD':1,
            'ZONE_CD':1,
        }
    }
]

#=====================
# Save Offers in a csv
#=====================
ls = []
cur = offers.aggregate(pipeline1, allowDiskUse=True)
for item in cur:
    ls.append(item)
df = pd.DataFrame(ls)
df.to_csv('data/corr.csv')

#==================
# Save MGP in a csv
#==================
done = False
drop = ['_id', 'Data', 'Ora']
ls = []
for doc in mgp.find({'Timestamp':{'$gte':1569877200.0}}):
    ts = doc['Timestamp']
    if not done:
        for k,v in doc.items():
            if 'PrezzoConv' in k or 'Coefficiente' in k or '_BID' in k or '_OFF' in k:
                drop.append(k)
        done = True
    ls.append(doc)
temp_df = pd.DataFrame(ls).drop(columns=drop)
temp_df.to_csv('data/mgp.csv')

#=======================
# Correlation Processing
#=======================
# Encode the cathegorical features
mgp = pd.read_csv('data/mgp.csv')
data = pd.read_csv('data/corr.csv')
for field in categ:
    data[field] = data[field].astype("category").cat.codes

# Join dataframes
col_ls = []
for (i,j) in mgp.iteritems():
    if 'Transito' in i:
        col_ls.append(i)
        data[i]=np.nan
for i in range(len(mgp)):
    for item in col_ls:
        data.loc[data['TIME']==mgp['Timestamp'][i], item] = mgp[item][i]
    print(f'{float(i/len(mgp))*100}%')

# Plot correlation
fig = plt.figure()
corr = data.dropna().corr()
sns.heatmap(corr)
plt.show()