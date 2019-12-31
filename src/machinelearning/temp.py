import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import time
from keras.callbacks import TensorBoard as tb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from src.common.mongo import MongoDB as Mongo

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from sklearn.decomposition import PCA
import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    from keras.models import Sequential, load_model
    from keras.layers import Dense, LSTM, Dropout, Bidirectional, BatchNormalization, Conv1D, MaxPooling1D, Flatten, RepeatVector, TimeDistributed
    from keras.optimizers import Adam
import pickle
import logging
import logging.config


logging.config.fileConfig('src/common/logging.conf')
logger = logging.getLogger(__name__)

"""
Alterinative model to test:


model.add(Conv1D(filters=64, kernel_size=3, input_shape=(X.shape[1], X.shape[2])))
model.add(Conv1D(filters=64, kernel_size=3))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(RepeatVector(X.shape[1]))
model.add(Bidirectional(LSTM(200, return_sequences=True,unroll=True)))
model.add(Dropout(0.3))
model.add(TimeDistributed(Dense(100)))
model.add(Dropout(0.3))
model.add(TimeDistributed(Dense(1)))
#model.compile(loss='mse', optimizer='adam')
"""

"""
TO IMPLEMENT LATER:

def getDifferences(df1, df2):
    df = pd.merge(df1, df2, left_index=True, right_index=True, how='outer', indicator='Exist')
    df = df.loc[df['Exist'] == 'right_only']
    
    for i in df.index:
        if i.strftime('%m-%Y') == datetime.now().strftime('%m-%Y'):
            
            return i
============================================
Descaling
============================================
Improve Model
============================================
Check if data are retrieved correctly from Mongo instead of using read_csv
============================================
Manage the Preprocess and Mongo folders

"""

class Preprocess():
    def __init__(self):
        self.X = []
        self.mat_list = []
        self.cnt = 1
        self.prev_hour = 0

    def loadData(self):
        #mng = Mongo(logger, USER, PASSWD)
        
        #bid, off = mng.operatorAggregate(start, operator)
        #mgp = mng.mgpAggregate(start)
        print('Loading Data')
        bid = pd.read_csv('bid.csv', index_col='Unnamed: 0')
        mgp = pd.read_csv('mgp.csv', index_col='Timestamp')
        bid.index = pd.to_datetime(bid.index)
        mgp.index = pd.to_datetime(mgp.index)

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
        )

        # Read MGP
        mgp = (
            pd
            .read_csv('mgp.csv', index_col='Timestamp')
            .iloc[1:]
        )

        # Merge dataframes
        merged = pd.merge(
            mgp, 
            off, 
            left_index=True, 
            right_index=True
        )
        print('Data Loaded')

        return merged

    def updateNew(self, new_sample, to_insert):
        self.mat_list.append({
            'matrix':new_sample,
            'filled_idx':-1
        })
        # Append the data referred to the current hour to all the matrix
        # by shifting its position
        for item in self.mat_list:
            item['filled_idx']+=1
            item['matrix'][item['filled_idx']] = to_insert

        # If a matrix is full move it to the X array
        for item in self.mat_list:
            if item['filled_idx'] == 23:
                self.X.append(self.mat_list.pop(0)['matrix'])  

        if self.cnt == 23:
            self.cnt = 0
        else:
            self.cnt+=1

    def prepareData(self, data):
        print('Prepared Data')
        for index in data.index:
            date = datetime.strptime(index, '%Y-%m-%d %H:%M:%S')
            # Create a new matrix representing 24h
            new_sample = np.zeros(
                (24, data.shape[1]),
                dtype=float
            )
            # If a hour is missing fill the row with zeros
            if date.hour != self.prev_hour:
                if date.hour == self.cnt:
                    to_insert = data.loc[index]
                else:
                    to_insert = np.zeros((1, 1,data.shape[1]), dtype=float)
                    while self.cnt!=date.hour:
                        self.updateNew(new_sample, to_insert)

                # Fix a dataset error
                if to_insert.shape[0]==2:
                    to_insert = to_insert.iloc[0]
                self.updateNew(new_sample, to_insert)
                self.prev_hour = date.hour
        self.X = np.asarray(self.X)
        print('Data Prepared')

        return np.asarray(self.X)
    
    def rollByLags(self, X):
        lag = -24
        y = np.roll(X, lag, axis=0)[:,:,-1]
        X, y = X[:lag,:,:], y[:lag,:]
        print('Rolled:')
        print(f'\tX:{X.shape}')
        print(f'\ty:{y.shape}')

        return X, y

class MGP():
    def __init__(self):
        self.history = None

    def descaleData(self, y_hat):
        with open('models/scaler.pkl', 'rb') as file:
            scalers = pickle.load(file)

        y_hat = y_hat.reshape(y_hat.shape[0],y_hat.shape[1])
        y_hat = scalers['y'].inverse_transform(y_hat)

        return y_hat

    def scaleData(self, X, *y):
        scalers = {
            'x':[]
        }

        if len(y)>0:
            y = y[0]
            split = .3    
            x_test = X[:int(X.shape[0]*split),:,:]
            x_train = X[int(X.shape[0]*split):,:,:]
            y_test = y[:int(y.shape[0]*split),:]
            y_train = y[int(y.shape[0]*split):,:]

            for i in range(X.shape[1]):
                scaler_x = StandardScaler()
                scaler_y = StandardScaler()
                
                scaler_x.fit(x_train[:,i,:])
                x_train[:,i,:] = scaler_x.transform(x_train[:,i,:])
                x_test[:,i,:] = scaler_x.transform(x_test[:,i,:])

                scalers['x'].append(scaler_x)
            
            scaler_y.fit(y_train)
            y_train = scaler_y.transform(y_train)
            y_test = scaler_y.transform(y_test)
            
            scalers['y'] = scaler_y

            y_test = y_test.reshape(y_test.shape[0],y_test.shape[1],1)
            y_train = y_train.reshape(y_train.shape[0],y_train.shape[1],1)

            with open('models/scaler.pkl', 'wb') as file:
                pickle.dump(scalers, file)
            print('Scaler saved')

            return x_train, y_train, x_test, y_test

        else:
            with open('models/scaler.pkl', 'rb') as file:
                scalers = pickle.load(file)
            print('Scaler Loaded')

            for i in range(X.shape[1]):
                scaler_x = scalers['x'][i]
                
                X[:,i,:] = scaler_x.transform(X[:,i,:])
            
            return X
    
    def trainModel(self, x_train, y_train, x_test, y_test):
        model = Sequential()

        model.add(
            LSTM(
                500, 
                input_shape=(x_train.shape[1], x_train.shape[2]), 
                return_sequences=True, 
                unroll=True
            )
        )
        model.add(Dropout(0.3))

        model.add(
            Bidirectional(
                LSTM(
                    200, 
                    input_shape=(x_train.shape[1], x_train.shape[2]),
                    unroll=True,
                    return_sequences=True, 
                )
            )
        )
        model.add(Dropout(0.3))

        model.add(TimeDistributed(Dense(100)))
        model.add(Dropout(0.3))

        model.add(TimeDistributed(Dense(1)))

        model.compile(loss='mse', optimizer='adam' , metrics = ['mae', 'mape'])
        print(model.summary())
        
        self.history = model.fit(
            x_train,
            y_train,
            epochs=1, 
            batch_size=50, 
            validation_data=(x_test,y_test), 
            shuffle=False,
        )

        model.save('models/my_model.h5') 
        print('Model Saved')
        return model
    
    def runModel(self, X):
        print('Predicting')
        model = load_model('models/my_model.h5')
        results = model.predict(X)
        print('Descaling')
        #results = self.descaleData(results)
        
        return results




# Train
p = Preprocess()
data = p.loadData()
X = p.prepareData(data)
X, y =  p.rollByLags(X)

mgp = MGP()
a,b,c,d = mgp.scaleData(X, y)
mgp.trainModel(
    a,b,c,d
)

# Validation
p = Preprocess()
data = p.loadData()
X = p.prepareData(data)
y = X[:,:,-1]

mgp = MGP()
y_hat = mgp.runModel(
    mgp.scaleData(X)
)
# Evaluate Model
plt.scatter(y, y_hat)
plt.plot(y,y, color='r')
plt.show()

plt.plot(y_hat[:,1])
plt.plot(y[:,1])
plt.show()