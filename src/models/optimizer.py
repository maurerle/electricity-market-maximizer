import pandas as pd 
from datetime import datetime, timedelta
import numpy as np
from influxdb import InfluxDBClient
TODAY = datetime.strptime('01/01/2018', '%d/%m/%Y')
target = 'IREN ENERGIA SPA'

client = InfluxDBClient(
    'localhost', 
    8086, 
    'root', 
    'root', 
    'PublicBids'
)
market = 'MGP'
res = client.query(f"show TAG values with key = op").raw
ops = pd.DataFrame(res['series'][0]['values']).drop(columns=0).values
ops = ops[:,0]
#ops = ops[3]

temp = client.query(f"select * from demandMGP where time = '{TODAY}'").raw
print(len(temp['series'][0]['values']))

from arima import Arima
discarded = []
considered = []
for op in ops:
    try:
        print(Arima(op).predict())
        considered.append(op)
    except KeyError:
        print('Error')

cnt = 0
for i in discarded:
    if i in op:
        print(i)
        cnt+=1

print(cnt)

print(len(considered))
