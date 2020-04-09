# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from intersection import intersection
from matplotlib import rcParams 


# +
def supCurve(df):
    curve = pd.DataFrame(columns=['OPS','P','Q'])
    cnt = 0
    for op in df['OPERATORE'].unique():
        new = pd.DataFrame(columns=['OPS','P','Q'])
        temp = df.where(df['OPERATORE']==op).dropna()
        new.loc[cnt] = [
            op,
            np.mean(temp['ENERGY_PRICE_NO']),
            np.sum(temp['QUANTITY_NO'])
        ]
        cnt+=1
        curve = pd.concat([curve, new], axis= 0)
    
    curve = curve.set_index('OPS')
    
    return curve

def demCurve(df):
    df['ENERGY_PRICE_NO'] = df['ENERGY_PRICE_NO'].replace(0, 3000)
    curve = pd.DataFrame(columns=['OPS','P','Q'])
    cnt = 0
    for op in df['OPERATORE'].unique():
        new = pd.DataFrame(columns=['OPS','P','Q'])
        temp = df.where(df['OPERATORE']==op).dropna()
        new.loc[cnt] = [
            op,
            np.mean(temp['ENERGY_PRICE_NO']),
            np.sum(temp['QUANTITY_NO'])
        ]
        cnt+=1
        curve = pd.concat([curve, new], axis= 0)
    
    curve = curve.set_index('OPS')
    
    return curve


# -

def loadData():
    data = (
        pd
        .read_csv('data/data.csv')
        .drop(columns=[
            'MARKET_CD', 
            'UNIT_REFERENCE_NO', 
            'MARKET_PARTECIPANT_XREF_NO',
            'BID_OFFER_DATE_DT',
            'TRANSACTION_REFERENCE_NO',
            'MERIT_ORDER_NO',
            'PARTIAL_QTY_ACCEPTED_IN',
            'ADJ_QUANTITY_NO',
            'ADJ_ENERGY_PRICE_NO',
            'GRID_SUPPLY_POINT_NO',
            'SUBMITTED_DT',
            'BALANCED_REFERENCE_NO',
        ])
    )

    #
    acc = (
        data
        .where(data['OPERATORE']!='Bilateralista')
        .where(data['STATUS_CD'].isin(['ACC', 'REJ']))
        .dropna()
    )
    off = acc.where(acc['PURPOSE_CD']=='OFF').dropna().drop(columns='PURPOSE_CD')
    bid = acc.where(acc['PURPOSE_CD']=='BID').dropna().drop(columns='PURPOSE_CD')
    
    return bid, off


def computeClearing(bid, off):
    sup = off.sort_values('P', ascending=True)
    dem = bid.sort_values('P', ascending=False)
    # Cumulative sums of quantity
    sup_cum = np.cumsum(sup['Q'])
    dem_cum = np.cumsum(dem['Q'])

    clearing = intersection(
        sup_cum.values, 
        sup.P.values, 
        dem_cum.values, 
        dem.P.values
    )[1][0]
    
    return clearing

# +
# Initialization
bid, off = loadData()

sup = supCurve(off)
dem = demCurve(bid)
# -



# +
target = 'IREN ENERGIA SPA'

# Determine the new curves
sup.loc[target] = [k, i]
dem.loc[target] = [w, j]
# Determine the clearing price
pun = computeClearing(dem, sup)
# Compute the profits
if sup.loc[target].P > pun:
    # Rejected bid for the supply
    Qsup = 0.0
else:
    # Accepted bid for the supply
    Qsup = sup.loc[target].Q
if dem.loc[target].P < pun:
    # Rejected bid for the demand
    Qdem = 0.0
else:
    # Accepted bid for the demand
    Qdem = dem.loc[target].Q

profit = (Qsup - Qdem)*pun
