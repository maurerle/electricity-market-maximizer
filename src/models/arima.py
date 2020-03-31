import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.graphics.tsaplots as tplot
from statsmodels.tsa.arima_model import ARIMA
from influxdb import InfluxDBClient
from sklearn.metrics import mean_squared_error
import json
import warnings


with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    # Line that is not converging
    #likev = mdf.profile_re(0, dist_low=0.1, dist_high=0.1)

#WINDOW = 60
WINDOW = 60
TODAY = datetime.strptime('01/01/2020', '%d/%m/%Y')
YDAY = TODAY + timedelta(days=-WINDOW)
#p_val = np.arange(0,6)
#q_val = np.arange(0,4)
p_val = np.arange(0,2)
q_val = np.arange(0,2)

H = 1


class Arima():
    def __init__(self, op):
        self.client = InfluxDBClient(
            'localhost', 
            8086, 
            'root', 
            'root', 
            'PublicBids'
        )
        self.op = op


    def manageIndexTest(self, data):
        start = TODAY + timedelta(days=-WINDOW)
        temp_date = pd.to_datetime(data['time'])
        data = (
            data
            .set_index(temp_date)
            .drop(columns=['time', 'OPS'])
        )
        idx = pd.date_range(start+ timedelta(days=+1), TODAY)
        data.index = pd.DatetimeIndex(data.index.date)
        data = (data.reindex(idx, fill_value=0))
        
        return data


    def manageIndexTrain(self, data):
        start = datetime.strptime('01/02/2017', '%d/%m/%Y')
        temp_date = pd.to_datetime(data['time'])
        data = (
            data
            .set_index(temp_date)
            .drop(columns=['time', 'OPS'])
        )
        idx = pd.date_range(start, TODAY)
        data.index = pd.DatetimeIndex(data.index.date)
        data = (data.reindex(idx, fill_value=0))
        
        return data


    def getDataTest(self, market):
        # Get the demand data from InfluxDB
        res = (
            self.client
            .query(
                f"SELECT * FROM demand{market} WHERE op='{self.op}' AND time <= '{TODAY}' and time > '{YDAY}'"
            )
            .raw
        )
        try:
            dem =(
                pd
                .DataFrame(
                    res['series'][0]['values'], 
                    columns = ['time', 'P', 'Q', 'OPS']
                )
            )
            dem = self.manageIndexTest(dem)
        except:
            dem = -1
        # Get the supply data from InfluxDB
        res = (
            self.client
            .query(
                f"SELECT * FROM supply{market} WHERE op='{self.op}' AND time <= '{TODAY}' and time > '{YDAY}'"
            )
            .raw
        )
        try:
            sup =(
                pd
                .DataFrame(
                    res['series'][0]['values'], 
                    columns = ['time', 'P', 'Q', 'OPS']
                )
            )
            sup = self.manageIndexTest(sup)
        except:
            sup = -1

        return dem, sup


    def getDataTrain(self, market):
        # Get the demand data from InfluxDB
        res = (
            self.client
            .query(
                f"SELECT * FROM demand{market} WHERE op='{self.op}' AND time <= '{TODAY}'"
            )
            .raw
        )

        dem =(
            pd
            .DataFrame(
                res['series'][0]['values'], 
                columns = ['time', 'P', 'Q', 'OPS']
            )
        )
        dem = self.manageIndexTrain(dem)
        # Get the supply data from InfluxDB
        res = (
            self.client
            .query(
                f"SELECT * FROM supply{market} WHERE op='{self.op}' AND time <= '{TODAY}'"
            )
            .raw
        )
        sup =(
            pd
            .DataFrame(
                res['series'][0]['values'], 
                columns = ['time', 'P', 'Q', 'OPS']
            )
        )
        sup = self.manageIndexTrain(sup)

        return dem, sup


    def runTest(self, data, label):
        """
        with open('config.json', 'r') as file:
            conf = json.loads(file.read())
        """
        [p, q] = [0, 0]
        X = data.values.astype('float32')
        model = ARIMA(
            X, 
            order=(p,1,q),
        )
        model_fit = model.fit(maxiter=200, disp=0, method='css')
        y = model_fit.forecast(H)[0][-1]
        if y < 0:
            y = 0
        return y 


    def runTrain(self, data, label):
        X = data.values
        mse = np.zeros((len(p_val), len(q_val)))
        for p in p_val:
            for q in q_val:
                y = []
                y_hat = []
                print(f'ARIMA({p},1,{q})')
                for i in range(X.shape[0]):
                    try:
                        train = X[i:i+WINDOW].astype('float32')

                        model = ARIMA(
                            train, 
                            order=(p,1,q),
                        )
                        model_fit = model.fit(maxiter=200, disp=0, method='css')
                        y.append(X[i+WINDOW-1])
                        y_hat.append(model_fit.forecast(H)[0][-1])
                    except:
                        pass
                try:
                    mse[p_val[p]][q_val[q]] = mean_squared_error(y, y_hat)
                except:
                    mse[p_val[p]][q_val[q]] = 1e7
        
        self.saveBest(mse, label)


    def saveBest(self, mse, label):
        mse = np.asarray(mse)
        best = np.amin(mse)
        loc = np.where(mse==best)
        loc = list(zip(loc[0], loc[1]))
        with open('config.json', 'r') as file:
            conf = json.loads(file.read())
        try:
            conf[self.op][label] = [int(loc[0][0]),int(loc[0][1])]
        except KeyError:
            obj = {
                "MGPpD": [],
                "MGPqD": [],
                "MGPpO": [],
                "MGPqO": [],
                "MIpD": [],
                "MIqD": [],
                "MIpO": [],
                "MIqO": [],
                "MSDpD": [],
                "MSDqD": [],
                "MSDpO": [],
                "MSDqO": []
            }
            conf[self.op] = obj
            conf[self.op][label] = [int(loc[0][0]),int(loc[0][1])]
        with open('config.json', 'w') as file:
            file.write(json.dumps(conf))


    def predict(self):
        pred = []
        for m in ['MGP', 'MI', 'MSD']:
            
            d, s = self.getDataTest(m)
            
            if isinstance(s, int):
                pred.append(-1.0)
                pred.append(-1.0)
            else:
                pred.append(self.runTest(s.P, f'{m}pO'))
                pred.append(self.runTest(s.Q, f'{m}qO'))
            
            if isinstance(d, int):
                pred.append(-1.0)
                pred.append(-1.0)
            else:
                pred.append(self.runTest(d.P, f'{m}pD'))
                pred.append(self.runTest(d.Q, f'{m}qD'))
    
        return np.asarray(pred)
    

    def train(self):
        for m in ['MGP', 'MI', 'MSD']:
            try:
                d, s = self.getDataTrain(m)
                self.runTrain(s.P, f'{m}pO')
                self.runTrain(s.Q, f'{m}qO')
                self.runTrain(d.P, f'{m}pD')
                self.runTrain(d.Q, f'{m}qD')
            except:
                mse = [[0, 0]]
                self.saveBest(mse, f'{m}pO')
                self.saveBest(mse, f'{m}qO')
                self.saveBest(mse, f'{m}pD')
                self.saveBest(mse, f'{m}qD')

"""
target = 'IREN ENERGIA SPA'
# Initialize the instance
arima = Arima(target)
# Train the model
arima.train()
# Predict the demanded Quantity
predictions = arima.predict()
print(predictions)
"""