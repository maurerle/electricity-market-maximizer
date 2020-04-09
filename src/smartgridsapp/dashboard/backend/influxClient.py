from influxdb import InfluxDBClient
import pandas as pd
import json
from datetime import datetime
from dateutil import relativedelta

now = datetime.now()
lastMonth = now - relativedelta.relativedelta(months=6)
lastMonth = int(datetime.timestamp(lastMonth)*1e9)
print(lastMonth)
client = InfluxDBClient(
    'localhost', 
    8086, 
    'root', 
    'root', 
    'PublicBids'
)

def getData(market, op):
    res = (
        client
        .query(f"SELECT * FROM demand{market} WHERE op = '{op}' AND time >= {lastMonth}")
        .raw
    )
    
    d_prices = []
    d_quants = []
    
    try:
        for val in res['series'][0]['values']:
            d_prices.append({'x':val[0], 'y':round(val[1],2)})
            d_quants.append({'x':val[0], 'y':round(val[2],2)})
    except:
        d_prices.append({'x':.0, 'y':.0})
        d_quants.append({'x':.0, 'y':.0})        
    res = (
        client
        .query(f"SELECT * FROM supply{market} WHERE op = '{op}' AND time >= {lastMonth}")
        .raw
    )
    
    o_prices = []
    o_quants = []

    try:
        for val in res['series'][0]['values']:
            o_prices.append({'x':val[0], 'y':round(val[1],2)})
            o_quants.append({'x':val[0], 'y':round(val[2],2)})
    except:
        o_prices.append({'x':.0, 'y':.0})
        o_quants.append({'x':.0, 'y':.0})

    return (
        json.dumps(d_prices),
        json.dumps(d_quants),
        json.dumps(o_prices),
        json.dumps(o_quants)
    )
    