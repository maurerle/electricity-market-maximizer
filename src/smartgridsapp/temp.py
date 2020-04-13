"""
from influxdb import InfluxDBClient
from datetime import datetime
from dateutil import relativedelta
CHOICE = []
now = datetime.now()
lastMonth = now - relativedelta.relativedelta(months=6)
lastMonth = int(datetime.timestamp(lastMonth)*1e9)

client = InfluxDBClient(
    'localhost',
    8086,
    'root',
    'root',
    'PublicBids'
)

for market in ['MGP', 'MI', 'MSD']:
    res = client.query(f"SELECT * FROM demand{market} WHERE time >= {lastMonth}").raw
    for val in res['series'][0]['values']:
        if val[3] not in CHOICE:
            CHOICE.append(val[3])

    res = client.query(f"SELECT * FROM supply{market} WHERE time >= {lastMonth}").raw
    for val in res['series'][0]['values']:
        if val[3] not in CHOICE:
            CHOICE.append(val[3])
"""

import numpy as np 
import matplotlib.pyplot as plt 

a = np.random.normal(0, 10, 1000)
tick = np.linspace(int(np.min(a)), int(np.max(a)), 20)
plt.hist(a, bins=tick)
plt.xticks(tick)
plt.show()