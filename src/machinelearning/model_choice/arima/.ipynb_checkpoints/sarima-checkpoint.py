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
    from statsmodels.tsa.statespace.sarimax import SARIMAX
plt.rcParams['figure.figsize'] = (10,7)

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


# -

def dickeyFuller(data, model):
    cutoff=0.00001
    # Perform Dickey-Fuller test:
    print(
        f'Dickey-Fuller Test {model}:'
    )
    p_val = adfuller(data['DLY_QTY'].dropna(), autolag='AIC')[1]
    if p_val < cutoff:
        print(f'\tp-value = {round(p_val,2)}. Stationary.')
    else:
        print(f'\tp-value = {round(p_val,2)}. Non-stationary.')

def inverseDifference(info, diffed):
    if info['order'] !=0:
        SIZE = len(diffed)
        inved = []
        LAGS = info[f"LAG{info['order']}"]
        for i in range(SIZE):
            if i<LAGS:
                inved.append(info[f"OBS_d{info['order']-1}"][i])
            else:
                inved.append(diffed[i-LAGS]+inved[i-LAGS])
        info['order']-=1
        fin = inverseDifference(info, inved)
        return fin
    
    else:
        return diffed


def statTest(data):
    # 0 order differentiation
    rolling_mean = data.rolling(window = 24).mean()
    rolling_std = data.rolling(window = 24).std()
    dickeyFuller(data, '0d')
    # 1 order differentiation
    rolling_mean_diff = (
        data
        .diff()
        .rolling(window = 24)
        .mean()
    )
    rolling_std_diff = (
        data
        .diff()
        .rolling(window = 24)
        .std()
    )
    dickeyFuller(data.diff(), '1d')
    # Plot the results
    plt.figure()
    plt.plot(data, label='Original')
    plt.plot(rolling_mean, label='Rolling mean')
    plt.plot(rolling_std, label='Rolling std')
    plt.plot(rolling_mean_diff, label='Rolling mean d1')
    plt.plot(rolling_std_diff, label='Rolling std d1')
    plt.xticks(ticks=np.arange(0,1000, 24), labels=np.arange(0,1000, 24))
    plt.xlabel('Date')
    plt.ylabel('Number of Rentals')
    plt.xlim(0)
    plt.grid()
    plt.legend()
    plt.show()
    
    
    
    # 0 order differentiation
    rolling_mean = data.rolling(window = 24).mean()
    rolling_std = data.rolling(window = 24).std()
    # 2 order differentiation
    rolling_mean_diff = (
        data
        .diff()
        .diff()
        .rolling(window = 24)
        .mean()
    )
    rolling_std_diff = (
        data
        .diff()
        .diff()
        .rolling(window = 24)
        .std()
    )
    dickeyFuller(data.diff().diff(), '2d')
    # Plot the results
    plt.figure()
    plt.plot(data, label='Original')
    plt.plot(rolling_mean, label='Rolling mean')
    plt.plot(rolling_std, label='Rolling std')
    plt.plot(rolling_mean_diff, label='Rolling mean d2')
    plt.plot(rolling_std_diff, label='Rolling std d2')
    plt.xticks(ticks=np.arange(0,1000, 24), labels=np.arange(0,1000, 24))
    plt.xlabel('Date')
    plt.ylabel('Number of Rentals')
    plt.xlim(0)
    plt.grid()
    plt.legend()
    plt.show()

# # Stationarity Test

statTest(off[:1000])

# # AutoCorrelation Function (ACF) 
# To determine q value

# +
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(15,10))
tplot.plot_acf(
    off, 
    ax[0],
    lags=40, 
    markersize=0,
    title='Original'
)
tplot.plot_acf(
    off.diff().dropna(), 
    ax[1],
    lags=40, 
    markersize=0,
    title='1st Order Differentiation'
)
tplot.plot_acf(
    off.diff().diff().dropna(), 
    ax[2],
    lags=40, 
    markersize=0,
    title='2nd Order Differentiation'
)

plt.show()
# -

# # Partial AutoCorrelation Function (PACF) 
# To determine p value

# +
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(15,10))
tplot.plot_pacf(
    off, 
    ax[0],
    lags=40, 
    markersize=0,
    title='Original'
)
tplot.plot_pacf(
    off.diff().dropna(), 
    ax[1],
    lags=40, 
    markersize=0,
    title='1st Order Differentiation'
)
tplot.plot_pacf(
    off.diff().diff().dropna(), 
    ax[2],
    lags=40, 
    markersize=0,
    title='2nd Order Differentiation'
)

plt.show()

# +
mse_df={
    '0':[], 
    '1':[],
    '2':[],
    '3':[],
    '4':[],
    '5':[],
    '6':[],
    '7':[],
    '8':[],
    '9':[],
    '10':[],
}
mpe_df={
    '0':[], 
    '1':[],
    '2':[],
    '3':[],
    '4':[],
    '5':[],
    '6':[],
    '7':[],
    '8':[],
    '9':[],
    '10':[],
}

min_mse = 1e7

p_val = np.arange(0,30,1)
d_val = [1, 2]
q_val = np.arange(0,6,1)
n = 24*7
end = 24

X = off.values
"""
X_1d = off.diff().dropna().values
X_2d = pd.DataFrame(X_1d).diff(24).dropna().values
train = X_2d[:n]
test = X_2d[n:n+end]
"""
train = X[:n]
test = X[n:n+end]

for d in d_val:
    for p in p_val:
        for q in q_val:

            """
            info = {
                'order':2,
                'LAG2':24,
                'LAG1':1,
                'OBS_d1':X_1d,
                'OBS_d0':X
            }
            """
            print(f'ARIMA({p},{d},{q})')
            try:
                history = [x for x in train]
                predictions = []
                for t in range(len(test)):
                    model = ARIMA(history, order=(p,d,q))
                    model_fit = model.fit(disp=0, method='css')
                    output = model_fit.forecast()
                    predictions.append(output[0])
                    history.append(test[t])

                #ccat = np.append(train, predictions)
                #pred = inverseDifference(info, ccat)[n:]
                #test_und = X[n:n+end]

                #mse = mean_squared_error(test_und, pred)
                #mpe = np.mean((test_und - pred) / test_und) * 100
                mse = mean_squared_error(test, predictions)
                mpe = np.mean((test - predictions) / test) * 100
            except:
                mse = np.nan
                mpe = np.nan
            #mse_df[f'{p}'].append(mse)
            #mpe_df[f'{p}'].append(mpe)

            print('Test MSE: %.3f' % mse)
            print('Test MPE: %.3f' % mpe)

            if mse < min_mse:
                min_mse = mse
                print('*****NEW MINIMUM DETECTED*****\n')
# -



















# +
p = 24
d = 2
q = 0

n = 24*7
end = 24 

X = off.values
X_1d = off.diff().dropna().values
X_2d = pd.DataFrame(X_1d).diff(24).dropna().values
#train = X_2d[:n]
#test = X_2d[n:n+end]
train = X[:n]
test = X[n:n+end]

info = {
    'order':2,
    'LAG2':24,
    'LAG1':1,
    'OBS_d1':X_1d,
    'OBS_d0':X
}

print(f'ARIMA({p},{d},{q})')
history = [x for x in train]
predictions = []

for t in range(len(test)):
    model = ARIMA(history, order=(p,d,q))
    model_fit = model.fit(disp=0, method='css')
    output = model_fit.forecast()
    predictions.append(output[0])
    history.append(test[t])
    #history = history[1:]
print(mean_squared_error(test,predictions))    
ccat = np.append(train, predictions)
# -

plt.plot(test)
plt.plot(predictions)

#model_fit.plot_predict(24*7, 24*7+24)
pred = model_fit.predict(24*7, 24*7+24)
plt.grid()
plt.plot(test, linestyle='-.', marker='*')
plt.plot(pred)

plt.plot(np.arange(550,672), train[550:672], marker='*', label = 'Original')
plt.plot(np.arange(550,673), model_fit.predict(550, 672), marker='o', label = 'Predictions')
plt.legend()

temp = inverseDifference(info, ccat)


plt.rcParams['figure.figsize'] = (20,15)
plt.plot(temp[n:], marker='o')
plt.plot(X[n:n+end])

plt.rcParams['figure.figsize'] = (20,15)
plt.plot(X[500:len(history)])
plt.plot(np.arange(n-500,n+end-500),temp[n:], color='r', marker='o', linewidth=0)
plt.xlim(0)
plt.grid(linestyle='-.')















# +
# Need Original data, diffed train and diffed predictions
PRED_S = len(predictions)
TRAIN_S = len(train)
inved = []
for i in range(TRAIN_S + PRED_S):
    if i<LAGS:
        inved.append(X[i])
    elif i<TRAIN_S+LAGS and i>=LAGS:
        inved.append(train[i-LAGS]+inved[i-LAGS])
    else:
        inved.append(predictions[i-TRAIN_S-LAGS]+inved[i-LAGS])
        
plt.rcParams['figure.figsize'] = (10,7)       
plt.plot(X[:len(history)], color='r')
plt.plot(inved, color='c', linestyle='-.')

# +
# Need Original data, diffed train and diffed predictions
temp = np.append(train,predictions)
PRED_S = len(predictions)
TRAIN_S = len(train)
inved = []
for i in range(TRAIN_S + PRED_S):
    if i<LAGS:
        inved.append(X[i])
    else:
        inved.append(temp[i-LAGS]+inved[i-LAGS])
  
plt.rcParams['figure.figsize'] = (10,7)       
plt.plot(X[:len(history)], color='r')
plt.plot(inved, color='c', linestyle='-.')
# -

plt.plot(X[:len(history)], color='r')
plt.plot(inved, color='c', linestyle='-.')


