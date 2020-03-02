# # 8 Day Forecasting Comparison

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# +
arima = pd.read_csv('arima_data_8days.csv')
lstm = pd.read_csv('lstm_data_8days.csv')
svr = pd.read_csv('svr_data_8days.csv')

arima = arima.drop(988)
lstm = lstm.drop(np.arange(0,46))
svr = svr.drop(np.arange(0,17))
# -

plt.plot(lstm.y.values[530:561], label='Observation', marker='o', color='C0', markersize=4)
plt.plot(lstm.y_hat.values[530:561], label='LSTM', marker='*', linestyle='-.', color='C1', markersize=4)
plt.plot(arima.y_hat.values[530:561], label='ARIMA(0,1,0)', marker='^', linestyle='--', color='C2', markersize=4)
plt.plot(svr.y_hat.values[530:561], label='SVR', marker='s', linestyle=':', color='C3', markersize=4)
plt.legend()
plt.xlabel('Days')
plt.ylabel('Average Daily Quantity')
plt.grid('--')
plt.xlim(0)
plt.savefig('comparison8days.png', transparent=True)
plt.close()


plt.scatter(lstm.y.values, lstm.y_hat.values, color='C3', s=10, alpha=.8, label='LSTM',marker='o')
plt.scatter(lstm.y.values, arima.y_hat.values, color='C1', s=9, alpha=.8, label='ARIMA(0,1,0)',marker='s')
plt.scatter(lstm.y.values, svr.y_hat.values, color='darkgreen', s=10, alpha=.8, label='SVR',marker='^')
plt.plot(lstm.y.values, lstm.y.values, color='r', linewidth=2)
plt.legend()
plt.xlabel('Observations')
plt.ylabel('Predictions')
plt.grid('--')
plt.xlim(0)
plt.savefig('scatter8days.png', transparent=True)
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

# +
arima = pd.read_csv('arima_data_1days.csv')
lstm = pd.read_csv('lstm_data_1days.csv')
svr = pd.read_csv('svr_data_1days.csv')

arima = arima.drop(988)
lstm = lstm.drop(np.arange(1041,1048))
lstm = lstm.drop(np.arange(0,53))
svr = svr.drop(np.arange(0,24))
# -

plt.plot(lstm.y.values[530:561], label='Observation', marker='o', color='C0', markersize=4)
plt.plot(lstm.y_hat.values[530:561], label='LSTM', marker='*', linestyle='-.', color='C1', markersize=4)
plt.plot(arima.y_hat.values[530:561], label='ARIMA(0,1,0)', marker='^', linestyle='--', color='C2', markersize=4)
plt.plot(svr.y_hat.values[530:561], label='SVR', marker='s', linestyle=':', color='C3', markersize=4)
plt.legend()
plt.xlabel('Days')
plt.ylabel('Average Daily Quantity')
plt.grid('--')
plt.xlim(0)
plt.savefig('comparison1days.png', transparent=True)
plt.close()

plt.scatter(lstm.y.values, lstm.y_hat.values, color='C3', s=10, alpha=.8, label='LSTM',marker='o')
plt.scatter(lstm.y.values, arima.y_hat.values, color='C1', s=9, alpha=.8, label='ARIMA(0,1,0)',marker='s')
plt.scatter(lstm.y.values, svr.y_hat.values, color='darkgreen', s=10, alpha=.8, label='SVR',marker='^')
plt.plot(lstm.y.values, lstm.y.values, color='r', linewidth=2)
plt.legend()
plt.xlabel('Observations')
plt.ylabel('Predictions')
plt.grid('--')
plt.xlim(0)
plt.savefig('scatter1days.png', transparent=True)
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

