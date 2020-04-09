# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from intersection import intersection
from matplotlib import rcParams 
from sklearn.metrics import roc_curve

rcParams.update({'font.size': 12})
rcParams.update({'figure.autolayout': True})
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
    
    curve = curve.sort_values('P', ascending=True)
    
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
    
    curve = curve.sort_values('P', ascending=False)
    
    return curve


# -

# # Demand/Supply  Curves: No Bilateral

# +
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

# +
sup = supCurve(off)
dem = demCurve(bid)
# Cumulative sums of quantity
sup_cum = np.cumsum(sup['Q'])
dem_cum = np.cumsum(dem['Q'])

clearing1 = intersection(sup_cum.values, sup.P.values, dem_cum.values, dem.P.values)[1][0]

# -
plt.figure()
plt.plot(sup_cum, sup.P, label='Supply', linewidth=2)
plt.plot(dem_cum, dem.P, label='Demand', linewidth=2)
plt.axhline(y=clearing1, linestyle='-.', color='k')
plt.ylim(0, 500)
plt.grid(linestyle='-.')
plt.legend()
plt.xlabel('Quantity [MWh]')
plt.ylabel('Price [\u20ac/MWh]')
plt.savefig('fig/CurvesNoB.png', transparent=True)

# +
obs_off = off['STATUS_CD'].replace('ACC', 1).replace('REJ',0)
obs_bid = bid['STATUS_CD'].replace('ACC', 1).replace('REJ',0)
pred_off = off.where(off['ENERGY_PRICE_NO']>clearing1)['STATUS_CD'].replace(['ACC', 'REJ'],0).fillna(1)
pred_bid = bid.where(bid['ENERGY_PRICE_NO']<clearing1)['STATUS_CD'].replace(['ACC', 'REJ'],0).fillna(1)
obs = pd.concat([obs_off, obs_bid], axis= 0)
pred = pd.concat([pred_off, pred_bid], axis= 0)

fpr, tpr, thresholds = roc_curve(obs, pred)
# -

# # Demand/Supply  Curves: Aggregated Bilateral

# +
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
    .where(data['STATUS_CD'].isin(['ACC', 'REJ']))
    .dropna()
)
off = acc.where(acc['PURPOSE_CD']=='OFF').dropna().drop(columns='PURPOSE_CD')
bid = acc.where(acc['PURPOSE_CD']=='BID').dropna().drop(columns='PURPOSE_CD')
# -

sup = supCurve(off)
dem = demCurve(bid)
# Cumulative sums of quantity
sup_cum = np.cumsum(sup['Q'])
dem_cum = np.cumsum(dem['Q'])
# Find Clearing Price
clearing2 = intersection(sup_cum.values, sup.P.values, dem_cum.values, dem.P.values)[1][0]

plt.figure()
plt.plot(sup_cum, sup.P, label='Supply', linewidth=2)
plt.plot(dem_cum, dem.P, label='Demand', linewidth=2)
plt.axhline(y=clearing2, linestyle='-.', color='k')
plt.ylim(0, 500)
plt.grid(linestyle='-.')
plt.legend()
plt.xlabel('Quantity [MWh]')
plt.ylabel('Price [\u20ac/MWh]')
plt.savefig('fig/CurvesB.png', transparent=True)

# +
obs_off = off['STATUS_CD'].replace('ACC', 1).replace('REJ',0)
obs_bid = bid['STATUS_CD'].replace('ACC', 1).replace('REJ',0)
pred_off = off.where(off['ENERGY_PRICE_NO']>clearing2)['STATUS_CD'].replace(['ACC', 'REJ'],0).fillna(1)
pred_bid = bid.where(bid['ENERGY_PRICE_NO']<clearing2)['STATUS_CD'].replace(['ACC', 'REJ'],0).fillna(1)
obs = pd.concat([obs_off, obs_bid], axis= 0)
pred = pd.concat([pred_off, pred_bid], axis= 0)

fprB, tprB, thresholds = roc_curve(obs, pred)

plt.figure()
plt.plot(fpr, tpr, label='NoBilateral', linewidth=2)
plt.plot(fprB, tprB, label='AggBilateral', linestyle='-.', linewidth=2)
plt.plot([0, 1], [0, 1], color='k', linestyle=':')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig('fig/roc.png', transparent=True)
# -




