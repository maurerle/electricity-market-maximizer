from influxdb import InfluxDBClient
from datetime import datetime, date
from dateutil import relativedelta
from src.arima import Arima

dummy_now = (
    date.today()- relativedelta.relativedelta(months=1)
).strftime('%d/%m/%Y')
TODAY = datetime.strptime(dummy_now, '%d/%m/%Y')
START = TODAY - relativedelta.relativedelta(days=60)
START = int(datetime.timestamp(START)*1e9)
OP_LIST = []

client = InfluxDBClient('172.28.5.1', 8086, 'root', 'root', 'PublicBids')

client.query(
    f"DELETE FROM predictions"
)

for market in ['MGP', 'MI', 'MSD']:
    res = client.query(f"SELECT * FROM demand{market} WHERE time >= {START}").raw
    for val in res['series'][0]['values']:
        if val[3] not in OP_LIST and "'" not in val[3]:
            OP_LIST.append(val[3])

    res = client.query(f"SELECT * FROM supply{market} WHERE time >= {START}").raw
    for val in res['series'][0]['values']:
        if val[3] not in OP_LIST and "'" not in val[3]:
            OP_LIST.append(val[3])

for op in OP_LIST:
    prediction = Arima(op, TODAY, client).predict()
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
print('Prediction Done')
