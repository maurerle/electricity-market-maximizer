# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


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
    data = (
        data
        .where(data['OPERATORE']!='Bilateralista')
        .where(data['STATUS_CD'].isin(['ACC', 'REJ']))
        .dropna()
    )
    off = (
        data
        .where(data['PURPOSE_CD']=='OFF')
        .drop(columns='PURPOSE_CD')
        .dropna()
    )
    bid = (
        data
        .where(data['PURPOSE_CD']=='BID')
        .drop(columns='PURPOSE_CD')
        .dropna()
    )
    
    return bid, off


# Initialization
bid, off = loadData()

sup = supCurve(off)
dem = demCurve(bid)

# +
from influxdb import InfluxDBClient

clientID = 'MGP'

client = InfluxDBClient('localhost', 8086, 'root', 'root', clientID)
if {'name': clientID} not in client.get_list_database():
    client.create_database(clientID)

# +
from datetime import datetime

for op in dem.index:
    body = [{
        'tags':{
            'op':op
        },
        'measurement':'Demand',
        'time':datetime.strptime('20170210','%Y%m%d'),
        'fields':{
            'Price':dem.loc[op].P,
            'Quantity':dem.loc[op].Q,
        }
    }]
    client.write_points(body, time_precision='h')

for op in sup.index:
    body = [{
        'tags':{
            'op':op
        },
        'measurement':'Demand',
        'time':datetime.strptime('20170210','%Y%m%d'),
        'fields':{
            'Price':sup.loc[op].P,
            'Quantity':sup.loc[op].Q,
        }
    }]
    client.write_points(body, time_precision='h')
# -


