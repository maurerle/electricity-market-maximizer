from influxdb import InfluxDBClient
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt


def getData(market):
    client = InfluxDBClient('localhost', 8086, 'root', 'root', clientID)
    # Demand
    res = client.query(f"SELECT * FROM demand{market} WHERE time = '{targetDay}'").raw
    dem =(
        pd
        .DataFrame(
            res['series'][0]['values'], 
            columns = ['time', 'P', 'Q', 'OPS']
        )
        .drop(columns=['time'])
        .set_index('OPS')
    )
    
    # Supply
    res = client.query(f"SELECT * FROM supply{market} WHERE time = '{targetDay}'").raw
    sup =(
        pd
        .DataFrame(
            res['series'][0]['values'], 
            columns = ['time', 'P', 'Q', 'OPS']
        )
        .drop(columns=['time'])
        .set_index('OPS')
    )
    
    return dem, sup


clientID = 'PublicBids'
targetDay = datetime.strptime('20200210','%Y%m%d')
targetOp = 'IREN ENERGIA SPA'

import numpy as np

MARKETS = ['MI1', 'MI2', 'MI3', 'MI4', 'MI5', 'MI6', 'MI7']

# +
MI = {
    'dem':[],
    'sup':[]
}

for m in MARKETS:
    d, s = getData(m)
    MI['dem'].append(d)
    MI['sup'].append(s)
MI['dem'][0]
MI['dem'][1]
MI['dem'][2]

mi_dQ = pd.concat((
    MI['dem'][0].Q, 
    MI['dem'][1].Q, 
    MI['dem'][2].Q,
    MI['dem'][3].Q, 
    MI['dem'][4].Q, 
    MI['dem'][5].Q,
    MI['dem'][6].Q
), axis=1).sum(axis=1)
mi_dP = pd.concat((
    MI['dem'][0].P, 
    MI['dem'][1].P, 
    MI['dem'][2].P,
    MI['dem'][3].P, 
    MI['dem'][4].P, 
    MI['dem'][5].P,
    MI['dem'][6].P
), axis=1).mean(axis=1)

dem = pd.DataFrame({
    'P':mi_dP,
    'Q':mi_dQ,
})
sup = pd.DataFrame

# +
dem.P = dem.P.replace(0, 3000)

sup = sup.sort_values('P', ascending=True)
dem = dem.sort_values('P', ascending=False)

# Cumulative sums of quantity
sup_cum = np.cumsum(sup['Q'])
dem_cum = np.cumsum(dem['Q'])


# -

plt.plot(dem_cum, dem.P)
plt.plot(sup_cum, sup.P)
plt.ylim(0,100)
plt.xlim(0,500000)


