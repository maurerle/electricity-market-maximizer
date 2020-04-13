import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import time
import os
from seaborn import heatmap, axes_style
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from matplotlib.colors import LinearSegmentedColormap
#from src.common.mongo import MongoDB as Mongo

# +
print('Loading Data')
bid = pd.read_csv('bid.csv', index_col='Unnamed: 0')
mgp = pd.read_csv('mgp.csv', index_col='Timestamp')
bid.index = pd.to_datetime(bid.index)
mgp.index = pd.to_datetime(mgp.index)

# Read Offerte Pubbliche 
off = (
    pd
    .read_csv('bid.csv', index_col='Unnamed: 0')
    .drop(
        columns = [
            #'DLY_AWD_QTY',
            'TYPE', 
            #'DLY_PRICE', 
            #'DLY_AWD_PRICE'
        ]
    )
)

# Read MGP
mgp = (
    pd
    .read_csv('mgp.csv', index_col='Timestamp')
    .iloc[1:]
)

# Merge dataframes
merged = pd.merge(
    mgp, 
    off, 
    left_index=True, 
    right_index=True
)


# +
dataset = merged.drop(columns=['DLY_QTY']).dropna()
cols=[x for x in dataset.head()]

corr = merged.corr()
corr = corr.drop(columns=cols).dropna()
corr = corr.where(abs(corr['DLY_QTY'])>.4).dropna()
corr = corr.drop(['DLY_AWD_QTY','DLY_AWD_PRICE', 'DLY_QTY'])
features = [x for x in corr.index]
features
