# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from intersection import intersection


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
    .read_csv('data.csv')
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
    .where(data['INTERVAL_NO']==1)
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

clearing = intersection(sup_cum.values, sup.P.values, dem_cum.values, dem.P.values)[1][0]
# -

plt.figure(figsize=(9,8))
plt.plot(sup_cum, sup.P, label='Supply', linewidth=2)
plt.plot(dem_cum, dem.P, label='Demand', linewidth=2)
plt.axhline(y=clearing, linestyle='-.', color='k')
plt.ylim(0, 200)
plt.xlim(0, 40000)
plt.grid(linestyle='-.')
plt.legend()
plt.xlabel('Quantity [MWh]')
plt.ylabel('Price [\u20ac/MWh]')

# +
accepted_sup = sup.where(sup.P<=clearing)
discarded_sup = sup.where(sup.P>clearing)
accepted_dem = dem.where(dem.P>=clearing)
discarded_dem = dem.where(dem.P<clearing)
acc_sup_cum = np.cumsum(sup['Q'])
acc_dem_cum = np.cumsum(dem['Q'])

plt.figure(figsize=(9,8))
plt.plot(acc_sup_cum, accepted_sup.P, label='ACC Supply', linewidth=2)
plt.plot(acc_dem_cum, accepted_dem.P, label='ACC Demand', linewidth=2)

plt.plot(acc_sup_cum, discarded_sup.P, label='REJ Supply', linestyle='--', linewidth=2)
plt.plot(acc_dem_cum, discarded_dem.P, label='REJ Demand', linestyle='--', linewidth=2)


plt.axhline(y=clearing, linestyle='-.', color='k')
plt.ylim(0, 200)
plt.xlim(0, 40000)
plt.grid()
plt.legend()
plt.xlabel('Quantity [MWh]')
plt.ylabel('Price [\u20ac/MWh]')
# -

# # Demand/Supply  Curves: Aggregated Bilateral

# +
data = (
    pd
    .read_csv('data.csv')
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
    #.where(data['OPERATORE']!='Bilateralista')
    .where(data['INTERVAL_NO']==1)
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
clearing = intersection(sup_cum.values, sup.P.values, dem_cum.values, dem.P.values)[1][0]

plt.figure(figsize=(9,8))
plt.plot(sup_cum, sup.P, label='Supply', linewidth=2)
plt.plot(dem_cum, dem.P, label='Demand', linewidth=2)
plt.axhline(y=clearing, linestyle='-.', color='k')
plt.ylim(0, 200)
plt.xlim(0, 40000)
plt.grid(linestyle='-.')
plt.legend()
plt.xlabel('Quantity [MWh]')
plt.ylabel('Price [\u20ac/MWh]')

# +
accepted_sup = sup.where(sup.P<=clearing)
discarded_sup = sup.where(sup.P>clearing)
accepted_dem = dem.where(dem.P>=clearing)
discarded_dem = dem.where(dem.P<clearing)
acc_sup_cum = np.cumsum(sup['Q'])
acc_dem_cum = np.cumsum(dem['Q'])

plt.figure(figsize=(9,8))
plt.plot(acc_sup_cum, accepted_sup.P, label='ACC Supply', linewidth=2)
plt.plot(acc_dem_cum, accepted_dem.P, label='ACC Demand', linewidth=2)

plt.plot(acc_sup_cum, discarded_sup.P, label='REJ Supply', linestyle='--', linewidth=2)
plt.plot(acc_dem_cum, discarded_dem.P, label='REJ Demand', linestyle='--', linewidth=2)


plt.axhline(y=clearing, linestyle='-.', color='k')
plt.ylim(0, 200)
plt.xlim(0, 40000)
plt.grid()
plt.legend()
plt.xlabel('Quantity [MWh]')
plt.ylabel('Price [\u20ac/MWh]')
# -


