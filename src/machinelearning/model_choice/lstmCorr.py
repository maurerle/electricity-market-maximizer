# +
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
#from src.common.mongo import MongoDB as Mongo
from sklearn.preprocessing import MinMaxScaler
from matplotlib import rcParams
import pickle

#rcParams.update({'font.size': 15})
# -

SPLIT = .7
TIMESTEPS = 7
MIN_CORR = .4


class Preprocess():
    def __init__(self, corr_limit):
        self.corr_limit = corr_limit

    def loadData(self):
        #mng = Mongo(logger, USER, PASSWD)
        
        #bid, off = mng.operatorAggregate(start, operator)
        #mgp = mng.mgpAggregate(start)
        
        # Read Offerte Pubbliche 
        off = (
            pd
            .read_csv('data/bid.csv', index_col='Unnamed: 0')
            .drop(
                columns = [
                    'DLY_AWD_QTY',
                    'TYPE', 
                    'DLY_PRICE', 
                    'DLY_AWD_PRICE'
                ]
            )
        )
        # Resampling
        off = off.set_index(pd.DatetimeIndex(off.index))
        off = off.resample('D').mean()
        # Get OFF(t) as feature
        label = np.roll(off['DLY_QTY'], -1, axis=0)
        # Discard the first values
        off = off[:-1]
        label = label[:-1]
        
        # Read MGP
        mgp = (
            pd
            .read_csv('data/mgp.csv', index_col='Timestamp')
            .iloc[1:]
        )
        mgp = mgp.set_index(pd.DatetimeIndex(mgp.index))
        mgp = mgp.resample('D').mean()
        mgp = mgp.iloc[:-(TIMESTEPS)]

        # Merge dataframes
        merged = pd.merge(
            mgp, 
            off, 
            left_index=True, 
            right_index=True
        )
        
        return merged, label
    
    def selectFeatures(self, dataframe, target):
        dataset = dataframe.copy()
        cols=[x for x in dataframe.head()]
        dataset['Target'] = target

        corr = dataset.corr()
        corr = corr.drop(columns=cols).dropna()
        corr = corr.where(abs(corr['Target'])>self.corr_limit).dropna()
        features = [x for x in corr.index if x != 'Target']
    
        
        return corr


# Train
p = Preprocess(MIN_CORR)
data, label = p.loadData()

corr = p.selectFeatures(data, label)
corr = corr.sort_values('Target', ascending=False)
corr = corr.drop('Target')
corr = corr.rename(
    {
        'DLY_QTY':'y(t)',
        'MGP_CNOR_Prezzo':'C-North Price',
        'MGP_AUS_Prezzo':'AT Price',
        'MGP_BSP_Prezzo':'SVN Coupling Price',
        'MGP_FRAN_Prezzo':'FR Price',
        'MGP_NORD_Prezzo':'North Price',
        'MGP_SLOV_Prezzo':'SVN Coupling Price',
        'MGP_SVIZ_Prezzo':'CH Coupling Price',
        'MGP_XAUS_Prezzo':'AT Coupling Price',
        'MGP_XFRA_Prezzo':'FR Coupling Price',
        'MGP_PUN_Prezzo':'National Price',
        'MGP_NAT_Prezzo':'NAT Price',
        'MGP_CSUD_Prezzo':'C-South Price',
        'MGP_COAC_Prezzo':'Corsica Price',
        'MGP_SARD_Prezzo':'Sardinia Coupling Price'
    }
)
corr['Target']*=100
rcParams.update({'figure.autolayout': True})
plt.figure()
#plt.plot(corr, marker='o')
plt.axhline(y=50, color='k', linestyle='-.', linewidth=1)
plt.axhline(y=45, color='k', linestyle='-.', linewidth=1)
plt.axhline(y=40, color='k', linestyle='-.', linewidth=1)
plt.stem(corr.index,corr,  bottom=-2)
plt.yticks([40,45,50,70,100])
plt.ylim(30)
plt.xlabel('Features')
plt.ylabel('Correlation [%]')
plt.xticks(rotation=90)
plt.savefig('fig/featuresCorrelation.png', transparent=True)