import pandas as pd 
from datetime import datetime, timedelta
import numpy as np
from influxdb import InfluxDBClient
from arima import Arima

#TODAY = datetime.strptime('01/01/2018', '%d/%m/%Y')
#target = 'IREN ENERGIA SPA'


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
res = client.query(f"show TAG values with key = op").raw
ops = pd.DataFrame(res['series'][0]['values']).drop(columns=0).values
ops = ops[:,0]

# Predict the strategy
for op in ops:
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