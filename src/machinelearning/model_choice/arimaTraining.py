import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import rcParams
from datetime import datetime
import numpy as np
#from src.common.mongo import MongoDB as Mongo
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
import statsmodels.graphics.tsaplots as tplot
from statsmodels.tsa.arima_model import ARIMA
import seaborn as sns
from matplotlib import rcParams

rcParams.update({'font.size': 12})
rcParams.update({'figure.autolayout': True})

PLIM = 6
QLIM = 4

WINDOW = 60

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
    rcParams.update({'figure.autolayout': True})
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
    plt.xlabel('Date')
    plt.ylabel('Average Daily Quantity')
    plt.xticks(rotation=90)
    plt.grid()
    plt.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.30))
    plt.savefig('fig/rollingD1.png', transparent=True)
    plt.close()
    rcParams.update({'figure.autolayout': False})


# # Load Data, Grid Search and Test Best

# +
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
# -

# ## Stationarity Test

statTest(df)

# ## AutoCorrelation Function (ACF) 
# To determine q value

# +
data = df
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15,10))
tplot.plot_acf(
    data, 
    ax[0],
    lags=40, 
    markersize=0,
    title='Original'
)
tplot.plot_acf(
    data.diff().dropna(), 
    ax[1],
    lags=40, 
    markersize=0,
    title='1st Order Differentiation'
)

plt.savefig('fig/acf.png', transparent=True)
plt.close()
# -

# ## Partial AutoCorrelation Function (PACF) 
# To determine p value

# +
data = df
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(15,10))
tplot.plot_pacf(
    data, 
    ax[0],
    lags=40, 
    markersize=0,
    title='Original'
)
tplot.plot_pacf(
    data.diff().dropna(), 
    ax[1],
    lags=40, 
    markersize=0,
    title='1st Order Differentiation'
)

plt.savefig('fig/pacf.png', transparent=True)
plt.close()
# -

# ## Grid Search
# ### Next Day

# +
h = 1

X = df.values
X = X[1:]

p_val = np.arange(0,PLIM)
q_val = np.arange(0,QLIM)

mse = np.zeros((len(p_val), len(q_val)))
mape = np.zeros((len(p_val), len(q_val)))
for p in p_val:
    for q in q_val:
        y = []
        y_hat = []
        print(f'ARIMA({p},1,{q})')
        for i in range(X.shape[0]):
            try:
                train = X[i:i+WINDOW]

                model = ARIMA(
                    train, 
                    order=(p,1,q),
                )
                model_fit = model.fit(maxiter=200, disp=0, method='css')
                y.append(X[i+WINDOW-1])
                y_hat.append(model_fit.forecast(h)[0][-1])
            except:
                pass
        mse[p_val[p]][q_val[q]] = mean_squared_error(y, y_hat)
# -

# ### MSE Heatmap

plt.figure(figsize=(6,8))
ax = sns.heatmap(mse, annot=True, fmt='.1f', cmap='YlGnBu', square=True, cbar_kws={'label':'MSE'}, vmin=0)
ax.set_ylim(PLIM,0)
ax.set_xlim(QLIM,0)
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_ylabel('Autoregressive Order (p)')
ax.set_xlabel('Moving Average Order (q)')
plt.savefig('fig/heatmap1day.png', transparent=True)
plt.close()

# ## Grid Search
# ### 8th Day

# +
h = 8

X = df.values
X = X[1:]

p_val = np.arange(0,PLIM)
q_val = np.arange(0,QLIM)

mse = np.zeros((len(p_val), len(q_val)))
mape = np.zeros((len(p_val), len(q_val)))
for p in p_val:
    for q in q_val:
        y = []
        y_hat = []
        print(f'ARIMA({p},1,{q})')
        for i in range(X.shape[0]):
            try:
                train = X[i:i+WINDOW]

                model = ARIMA(
                    train, 
                    order=(p,1,q),
                )
                model_fit = model.fit(maxiter=200, disp=0, method='css')
                y.append(X[i+WINDOW-1])
                y_hat.append(model_fit.forecast(h)[0][-1])
            except:
                pass
        mse[p_val[p]][q_val[q]] = mean_squared_error(y, y_hat)
# -

# ### MSE Heatmap

plt.figure(figsize=(6,8))
ax = sns.heatmap(mse, annot=True, fmt='.1f', cmap='YlGnBu', square=True, cbar_kws={'label':'MSE'}, vmin=0)
ax.set_ylim(PLIM,0)
ax.set_xlim(QLIM,0)
ax.invert_yaxis()
ax.invert_xaxis()
ax.set_ylabel('Autoregressive Order (p)')
ax.set_xlabel('Moving Average Order (q)')
plt.savefig('fig/heatmap8day.png', transparent=True)
plt.close()


