# # 8 Day Forecasting Comparison

import pandas as pd
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rcParams.update({'font.size': 12})
rcParams.update({'figure.autolayout': True})

arima = pd.read_csv('data/arima8days.csv')
lstm = pd.read_csv('data/lstm8days.csv')
svr = pd.read_csv('data/svr8days.csv')

arima = arima.drop(988)
lstm = lstm.drop(np.arange(0,46))
svr = svr.drop(np.arange(0,45))

plt.plot(lstm.y.values[530:561], label='Observation', marker='o', color='C0', markersize=4)
plt.plot(lstm.y_hat.values[530:561], label='LSTM', marker='*', linestyle='-.', color='C1', markersize=4)
plt.plot(arima.y_hat.values[530:561], label='ARIMA(0,1,0)', marker='^', linestyle='--', color='C2', markersize=4)
plt.plot(svr.y_hat.values[530:561], label='SVR', marker='s', linestyle=':', color='C3', markersize=4)
plt.legend()
plt.xlabel('Days')
plt.ylabel('Daily Quantity [MWh]')
plt.grid('--')
plt.xlim(0)
plt.savefig('fig/comparison8days.png', transparent=True)
plt.close()




plt.scatter(lstm.y.values, lstm.y_hat.values, color='C3', s=10, alpha=.8, label='LSTM',marker='o')
plt.scatter(lstm.y.values, arima.y_hat.values, color='C1', s=9, alpha=.8, label='ARIMA(0,1,0)',marker='s')
plt.scatter(lstm.y.values, svr.y_hat.values, color='darkgreen', s=10, alpha=.8, label='SVR',marker='^')
plt.plot(lstm.y.values, lstm.y.values, color='r', linewidth=2)
plt.legend()
plt.xlabel('Observations [MWh]')
plt.ylabel('Predictions [MWh]')
plt.grid('--')
plt.xlim(0)
plt.savefig('fig/scatter8days.png', transparent=True)
plt.close()

# +
print('ARIMA(0,1,0)')
mse = mean_squared_error(lstm.y.values, arima.y_hat.values)
mape = np.mean(np.abs((lstm.y.values - arima.y_hat.values) / lstm.y.values)) * 100
mae = mean_absolute_error(lstm.y.values, arima.y_hat.values)
r2 = r2_score(lstm.y.values, arima.y_hat.values)
print(f'\tMSE: {round(mse,2)}')
print(f'\tMAE: {round(mae,2)}')
print(f'\tMAPE: {round(mape,2)}%')
print(f'\tR2: {round(r2,2)}')
print('LSTM')
mse = mean_squared_error(lstm.y.values, lstm.y_hat.values)
mape = np.mean(np.abs((lstm.y.values - lstm.y_hat.values) / lstm.y.values)) * 100
mae = mean_absolute_error(lstm.y.values, lstm.y_hat.values)
r2 = r2_score(lstm.y.values, lstm.y_hat.values)
print(f'\tMSE: {round(mse,2)}')
print(f'\tMAE: {round(mae,2)}')
print(f'\tMAPE: {round(mape,2)}%')
print(f'\tR2: {round(r2,2)}')

print('SVR')
mse = mean_squared_error(lstm.y.values, svr.y_hat.values)
mape = np.mean(np.abs((lstm.y.values - svr.y_hat.values) / lstm.y.values)) * 100
mae = mean_absolute_error(lstm.y.values, svr.y_hat.values)
r2 = r2_score(lstm.y.values, svr.y_hat.values)
print(f'\tMSE: {round(mse,2)}')
print(f'\tMAE: {round(mae,2)}')
print(f'\tMAPE: {round(mape,2)}%')
print(f'\tR2: {round(r2,2)}')

# -

# # 1 Day Forecasting Comparison

arima = pd.read_csv('data/arima1days.csv')
lstm = pd.read_csv('data/lstm1days.csv')
svr = pd.read_csv('data/svr1days.csv')

arima = arima.drop(988)
lstm = lstm.drop(np.arange(1041,1048))
lstm = lstm.drop(np.arange(0,53))
svr = svr.drop(np.arange(0,52))

plt.plot(lstm.y.values[530:561], label='Observation', marker='o', color='C0', markersize=4)
plt.plot(lstm.y_hat.values[530:561], label='LSTM', marker='*', linestyle='-.', color='C1', markersize=4)
plt.plot(arima.y_hat.values[530:561], label='ARIMA(0,1,0)', marker='^', linestyle='--', color='C2', markersize=4)
plt.plot(svr.y_hat.values[530:561], label='SVR', marker='s', linestyle=':', color='C3', markersize=4)
plt.legend()
plt.xlabel('Days')
plt.ylabel('Daily Quantity [MWh]')
plt.grid('--')
plt.xlim(0)
plt.savefig('fig/comparison1days.png', transparent=True)
plt.close()

plt.scatter(lstm.y.values, lstm.y_hat.values, color='C3', s=10, alpha=.8, label='LSTM',marker='o')
plt.scatter(lstm.y.values, arima.y_hat.values, color='C1', s=9, alpha=.8, label='ARIMA(0,1,0)',marker='s')
plt.scatter(lstm.y.values, svr.y_hat.values, color='darkgreen', s=10, alpha=.8, label='SVR',marker='^')
plt.plot(lstm.y.values, lstm.y.values, color='r', linewidth=2)
plt.legend()
plt.xlabel('Observations [MWh]')
plt.ylabel('Predictions [MWh]')
plt.grid('--')
plt.xlim(0)
plt.savefig('fig/scatter1days.png', transparent=True)
plt.close()

# +
print('ARIMA(0,1,0)')
mse = mean_squared_error(lstm.y.values, arima.y_hat.values)
mape = np.mean(np.abs((lstm.y.values - arima.y_hat.values) / lstm.y.values)) * 100
mae = mean_absolute_error(lstm.y.values, arima.y_hat.values)
r2 = r2_score(lstm.y.values, arima.y_hat.values)
print(f'\tMSE: {round(mse,2)}')
print(f'\tMAE: {round(mae,2)}')
print(f'\tMAPE: {round(mape,2)}%')
print(f'\tR2: {round(r2,2)}')

print('LSTM')
mse = mean_squared_error(lstm.y.values, lstm.y_hat.values)
mape = np.mean(np.abs((lstm.y.values - lstm.y_hat.values) / lstm.y.values)) * 100
mae = mean_absolute_error(lstm.y.values, lstm.y_hat.values)
r2 = r2_score(lstm.y.values, lstm.y_hat.values)
print(f'\tMSE: {round(mse,2)}')
print(f'\tMAE: {round(mae,2)}')
print(f'\tMAPE: {round(mape,2)}%')
print(f'\tR2: {round(r2,2)}')
print('SVR')
mse = mean_squared_error(lstm.y.values, svr.y_hat.values)
mape = np.mean(np.abs((lstm.y.values - svr.y_hat.values) / lstm.y.values)) * 100
mae = mean_absolute_error(lstm.y.values, svr.y_hat.values)
r2 = r2_score(lstm.y.values, svr.y_hat.values)
print(f'\tMSE: {round(mse,2)}')
print(f'\tMAE: {round(mae,2)}')
print(f'\tMAPE: {round(mape,2)}%')
print(f'\tR2: {round(r2,2)}')

