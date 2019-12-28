import pandas as pd
from src.common.dataProcessing import DataProcessing
import logging
import logging.config
import numpy as np 
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, BatchNormalization, GRU, ELU
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import pprint

logging.config.fileConfig('src/common/logging.conf')
logger = logging.getLogger(__name__)


def scaling(X, y):
    global scaler
    
    data = np.concatenate((X,y),axis=1)
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data)

    y = data[data.shape[1]-1].to_numpy()
    X = data[range(data.shape[1]-1)].to_numpy()
    X = X.reshape(X.shape[0], 1, X.shape[1])

    return X, y

def descaling(x_test, y_test, results):
    # Descaling
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[2])
    y_test = y_test.reshape(len(y_test), 1)
    results = results.reshape(len(results), 1)

    dset1 = np.concatenate((x_test, y_test),axis=1)
    dset2 = np.concatenate((x_test, results), axis=1)

    dset1 = scaler.inverse_transform(dset1)
    dset2 = scaler.inverse_transform(dset2)


    y = dset1[:,dset2.shape[1]-1]
    y_hat = dset2[:,dset2.shape[1]-1]

    return y, y_hat


class MGP():
    def __init__ (self, user, passwd):
        self.user = user
        self.passwd = passwd

    def createSet(self):
        mongo = DataProcessing(logger, self.user, self.passwd)
        
        dataset = mongo.merge(
            mongo.mgpAggregate(1543615200.0),
            mongo.operatorAggregate('IREN ENERGIA SPA', 'OffertePubbliche')
        )
        dataset.to_csv('datasetTest.csv')

    def manage(self):
        lag = 1
        data = pd.read_csv('datasetTest.csv')
        data = data.drop(columns=['TYPE', 'Unnamed: 0'])
        # Create dataset before the scaling
        new_data=data.shift(periods=-lag, fill_value=0)
        y = new_data['DLY_QTY'].iloc[:len(new_data.index)-lag,].to_numpy(dtype=float)
        
        new_data = new_data.drop(columns=['DLY_QTY','DLY_PRICE','DLY_AWD_QTY','DLY_AWD_PRICE'])
        to_rename = {}
        for item in new_data.columns:
            to_rename[item] =item + '_NXT'
        new_data = new_data.rename(columns=to_rename)
        data1 = data['DLY_QTY']
        data = data.join(new_data)
        #data = data['DLY_QTY']
        pca = PCA(n_components=30, whiten=True)
        
        X = data.iloc[:len(data.index)-lag,].to_numpy(dtype=float)
        #X = X.reshape(X.shape[0], 1)
        X = pca.fit_transform(X)
        y = y.reshape(len(y), 1)

        print(X.shape)
        print(y.shape)

        X, y = scaling(X, y)

        x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state = 4)

        model = Sequential()
        
        # MAPE 9.7%
        """
        model.add(LSTM(80, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Bidirectional(LSTM(50, input_shape=(X.shape[1], X.shape[2]))))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Dense(40))
        model.add(Dropout(0.1))
        
        model.add(Dense(1))
        
        """
        # MAPE 9.6%
        model.add(LSTM(80, input_shape=(X.shape[1], X.shape[2]), return_sequences=True, unroll=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Bidirectional(LSTM(50, input_shape=(X.shape[1], X.shape[2]))))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(40))
        model.add(Dropout(0.1))
        
        model.add(Dense(1))
        
        model.compile(loss='mse', optimizer='adam')
        print(model.summary())
        
        history = model.fit(x_train,y_train,epochs=90, batch_size=50, validation_data=(x_test,y_test), shuffle=False)

        results = model.predict(x_test)

        y_test, y_hat_test = descaling(x_test, y_test, results)

        rmse = sqrt(mean_squared_error(y_test, y_hat_test))
        r2 = r2_score(y_test, y_hat_test)
        mae = mean_absolute_error(y_test, y_hat_test)
        mape = np.mean(np.abs((y_test - y_hat_test) / y_test)) * 100
        print(f'RMSE Test:  {rmse}')
        print(mae)
        print(f'MAPE: {mape}%')
        print(f'R2 Test:  {r2}')

        plt.scatter(np.arange(y_hat_test.shape[0]), y_hat_test, s=1, color='r')
        plt.plot(y_test, linewidth=.5)
        plt.show()

        #fig = plt.figure()
        plt.xlabel('y')
        plt.ylabel('y_hat')
        plt.scatter(y_test,y_hat_test, s=1, label = 'y vs. y_hat')
        plt.plot(y_test, y_test, 'r', label = 'y = y')
        plt.legend()
        plt.show()

        plt.ylabel('MSE')
        plt.xlabel("Epochs")
        plt.plot(history.history['loss'], label = 'Test')
        plt.plot(history.history['val_loss'], label = 'Train')
        plt.legend()
        plt.show()






        pca = PCA(n_components=30, whiten=True)
        
        X = data.iloc[:len(data.index)-lag,].to_numpy(dtype=float)
        #X = X.reshape(X.shape[0], 1)
        X = pca.fit_transform(X)
        y = y.reshape(len(y), 1)

        print(X.shape)
        print(y.shape)

        X, y = scaling(X, y)
        results = model.predict(X)

        y_test, y_hat_test = descaling(X, y, results)

        plt.plot(y_test, linewidth=.5)
        plt.plot(y_hat_test, linewidth=.4)
        plt.show()

class MI():
    def __init__ (self):
        pass

class MSD():
    def __init__ (self):
        pass