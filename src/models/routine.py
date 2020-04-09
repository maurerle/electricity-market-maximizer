import pandas as pd 
from datetime import datetime, timedelta
import numpy as np
from influxdb import InfluxDBClient
from arima import Arima
from geneticModule import Genetic
from datetime import datetime
from dateutil import relativedelta

now = datetime.now()
lastMonth = now - relativedelta.relativedelta(months=4)
lastMonth = int(datetime.timestamp(lastMonth)*1e9)
CHOICE = []
def predict_strategies():
    # Get the operator list
    client = InfluxDBClient(
        'localhost', 
        8086, 
        'root', 
        'root', 
        'PublicBids'
    )

    client.query(
        f"DELETE FROM predictions"
    )
    # Get the active Operator
    for market in ['MGP', 'MI', 'MSD']:
        res = client.query(f"SELECT * FROM demand{market} WHERE time >= {lastMonth}").raw
        for val in res['series'][0]['values']:
            if val[3] not in CHOICE:
                CHOICE.append(val[3])

        res = client.query(f"SELECT * FROM supply{market} WHERE time >= {lastMonth}").raw
        for val in res['series'][0]['values']:
            if val[3] not in CHOICE:
                CHOICE.append(val[3])

    # Predict the strategy
    for op in CHOICE:
        print(op)
        if not "'" in op:
            prediction = Arima(op).predict()
            # Upload the prediction to the dataBase
            body = [{
                'tags':{
                    'op':op
                },
                'measurement':f'predictions',
                'fields':{
                    'MGPpO':prediction[0],
                    'MGPqO':prediction[1],
                    'MGPpD':prediction[2],
                    'MGPqD':prediction[3],
                    'MIpO':prediction[4],
                    'MIqO':prediction[5],
                    'MIpD':prediction[6],
                    'MIqD':prediction[7],
                    'MSDpO':prediction[8],
                    'MSDqO':prediction[9],
                    'MSDpD':prediction[10],
                    'MSDqD':prediction[11],
                }
            }]
            client.write_points(body)

            print()

def optimize(op):
    client = InfluxDBClient(
        'localhost', 
        8086, 
        'root', 
        'root', 
        'PublicBids'
    )

    res = client.query(f"SELECT * FROM predictions").raw

    predictions = (
        pd
        .DataFrame(
            res['series'][0]['values'], 
            columns = res['series'][0]['columns']
        )
        .drop(columns='time')
        .set_index('op')
    )

    Genetic(op, predictions).run()

#predict_strategies()
#optimize('S.E.F. SRL')
# Get the operator list
client = InfluxDBClient(
    'localhost', 
    8086, 
    'root', 
    'root', 
    'PublicBids'
)

# Get the active Operator
for market in ['MGP', 'MI', 'MSD']:
    res = client.query(f"SELECT * FROM demand{market} WHERE time >= {lastMonth}").raw
    for val in res['series'][0]['values']:
        if val[3] not in CHOICE:
            CHOICE.append(val[3])

    res = client.query(f"SELECT * FROM supply{market} WHERE time >= {lastMonth}").raw
    for val in res['series'][0]['values']:
        if val[3] not in CHOICE:
            CHOICE.append(val[3])

#predict_strategies()

for op in CHOICE:
    if not "'" in op:
        print(op)
        optimize(op)


