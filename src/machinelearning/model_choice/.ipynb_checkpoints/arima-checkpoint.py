import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
#from src.common.mongo import MongoDB as Mongo
from statsmodels.tsa.stattools import adfuller
import statsmodels.graphics.tsaplots as tplot
from statsmodels.tsa.arima_model import ARIMA

WINDOW = 60

# Read Offerte Pubbliche 
off = (
    pd
    .read_csv('data/bid.csv', index_col='Unnamed: 0')
    .drop(
        columns = [
            'DLY_AWD_QTY',
            'TYPE', 
            'DLY_PRICE', 
            'DLY_AWD_PRICE'
        ]
    )
    .dropna()
)
df = off 
df = df.set_index(pd.DatetimeIndex(off.index))
df = df.resample('D').sum()

# # Next Day Validation
print('Performing 1 day validation')
h = 1

X = df.values
X = X[1:]

y = []
y_hat = []

for i in range(X.shape[0]):
    try:
        train = np.copy(X[i:i+WINDOW])

        model = ARIMA(
            train, 
            order=(0,1,0),
        )
        model_fit = model.fit(maxiter=200, disp=0, method='css')
        y.append(X[i+WINDOW-1])
        y_hat.append(model_fit.forecast(h)[0][-1])
    except:
        pass
to_save = {
    'y':y,
    'y_hat':y_hat
}
pd.DataFrame(to_save).to_csv('data/arima1days.csv')

# # 8th Day Validation
print('Performing 8 days validation')
h = 8

X = df.values
X = X[1:]

y = []
y_hat = []

for i in range(X.shape[0]):
    try:
        train = np.copy(X[i:i+WINDOW])

        model = ARIMA(
            train, 
            order=(0,1,0),
        )
        model_fit = model.fit(maxiter=200, disp=0, method='css')
        y.append(X[i+WINDOW-1])
        y_hat.append(model_fit.forecast(h)[0][-1])
    except:
        pass
# -
to_save = {
    'y':y,
    'y_hat':y_hat
}
pd.DataFrame(to_save).to_csv('data/arima8days.csv')


