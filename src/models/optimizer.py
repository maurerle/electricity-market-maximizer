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

from arima import Arima
for op in ops:
    print(op)
    print(Arima(op).predict())
    print()