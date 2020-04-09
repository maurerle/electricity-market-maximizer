import pandas as pd 
from influxdb import InfluxDBClient
from datetime import datetime
from dateutil import relativedelta
import numpy as np
from .arima import ArimaV2
from .genetic import Genetic

TODAY = datetime.strptime('15/03/2020', '%d/%m/%Y')
START = TODAY - relativedelta.relativedelta(days=60)
START = int(datetime.timestamp(START)*1e9)
OP_LIST = []


def optimize(target, limit):
    
    client = InfluxDBClient('localhost', 8086, 'root', 'root', 'PublicBids')


    for market in ['MGP', 'MI', 'MSD']:
        res = client.query(f"SELECT * FROM demand{market} WHERE time >= {START}").raw
        for val in res['series'][0]['values']:
            if val[3] not in OP_LIST and "'" not in val[3]:
                OP_LIST.append(val[3])

        res = client.query(f"SELECT * FROM supply{market} WHERE time >= {START}").raw
        for val in res['series'][0]['values']:
            if val[3] not in OP_LIST and "'" not in val[3]:
                OP_LIST.append(val[3])

    df = pd.DataFrame(columns=[
        'MGPpO', 'MGPqO', 'MGPpD', 'MGPqD', 'MIpO', 'MIqO',
       'MIpD', 'MIqD', 'MSDpO', 'MSDqO', 'MSDpD', 'MSDqD', 'op'
    ])

    for op in OP_LIST:
        arima = ArimaV2(op, TODAY)
        pred = arima.predict()
        pred = np.append(pred, op)
        temp = pd.DataFrame([pred], columns=[
       'MGPpO', 'MGPqO', 'MGPpD', 'MGPqD', 'MIpO', 'MIqO',
       'MIpD', 'MIqD', 'MSDpO', 'MSDqO', 'MSDpD', 'MSDqD','op'
        ])
        df = df.append(temp, ignore_index = True)
    
    df.to_csv('temp.csv')
    
    data = pd.read_csv('temp.csv').set_index('op')
    
    genetic = Genetic(target, data, TODAY, limit)

    return genetic.run()

#optimize('IREN ENERGIA SPA', 1500.0)