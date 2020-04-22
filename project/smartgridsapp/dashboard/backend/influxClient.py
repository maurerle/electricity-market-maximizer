from influxdb import InfluxDBClient
import pandas as pd
import json
from datetime import datetime, date
from dateutil import relativedelta

def getData(market, op):
    """Get data from InfluxDB.
    
    Parameters
    ----------
    market : str
        Market type
    op : str
        Operator
    
    Returns
    -------
    JSON str, JSON str, JSON str, JSON str 
        Demanded and offered prices and quantities
    """
    dummy_now = (
        date.today()- relativedelta.relativedelta(months=1)
    ).strftime('%d/%m/%Y')
    TODAY = datetime.strptime(dummy_now, '%d/%m/%Y')
    START = TODAY - relativedelta.relativedelta(days=60)
    START = int(datetime.timestamp(START)*1e9)

    client = InfluxDBClient(
        '172.28.5.1', 
        8086, 
        'root', 
        'root', 
        'PublicBids'
    )

    res = (
        client
        .query(f"SELECT * FROM demand{market} WHERE op = '{op}' AND time >= {START}")
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
        .query(f"SELECT * FROM supply{market} WHERE op = '{op}' AND time >= {START}")
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
    