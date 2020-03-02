import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import time
#from src.common.mongo import MongoDB as Mongo
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from statsmodels.tsa.stattools import adfuller
import statsmodels.graphics.tsaplots as tplot
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=RuntimeWarning)
    from statsmodels.tsa.arima_model import ARIMA

# +
bid = pd.read_csv('bid.csv', index_col='Unnamed: 0')
bid.index = pd.to_datetime(bid.index)

# Read Offerte Pubbliche 
off = (
    pd
    .read_csv('bid.csv', index_col='Unnamed: 0')
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
df = df.resample('D').mean()

# # Next Day Validation

# +
WINDOW = 60
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
        print(f'Out of Range: {i}-th iteration')
# -

mse = mean_squared_error(y, y_hat)
mape = np.mean(np.abs((np.asarray(y) - np.asarray(y_hat))) / np.asarray(y)) * 100
print(f'MSE:{mse}')
print(f'MAPE:{mape}')
# Plot the results
plt.figure(figsize=(12,8))
plt.plot(y_hat, color='g', label='Predictions')
plt.plot(y, color='r', label='Observations')
plt.legend()
plt.show()

to_save = {
    'y':y,
    'y_hat':y_hat
}
pd.DataFrame(to_save).to_csv('arima_data_1days.csv')

# # 8th Day Validation

# +
WINDOW = 60
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
        print(f'Out of Range: {i}-th iteration')
# -

mse = mean_squared_error(y, y_hat)
mape = np.mean(np.abs((np.asarray(y) - np.asarray(y_hat))) / np.asarray(y)) * 100
print(f'MSE:{mse}')
print(f'MAPE:{mape}')
# Plot the results
plt.figure(figsize=(12,8))
plt.plot(y_hat, color='g', label='Predictions')
plt.plot(y, color='r', label='Observations')
plt.legend()
plt.show()

to_save = {
    'y':y,
    'y_hat':y_hat
}
pd.DataFrame(to_save).to_csv('arima_data_8days.csv')


