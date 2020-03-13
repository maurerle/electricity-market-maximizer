import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import time
from keras.callbacks import TensorBoard as tb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
#from src.common.mongo import MongoDB as Mongo

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
        with open('scalery.pkl', 'rb') as file:
            scaler_y = pickle.load(file)

        y_hat_r = y_hat.reshape(y_hat.shape[0],y_hat.shape[1])
        y_hat_d = scaler_y.inverse_transform(y_hat_r)

        return y_hat_d
    
    def scaleData(self, X_toScale, *y):
        split = .7

        if len(y)>0:
            y = y[0]

            # Initialize the training and test items
            x_test = X_toScale[int(X_toScale.shape[0]*split):,:,:]
            x_train = X_toScale[:int(X_toScale.shape[0]*split),:,:]
            y_test = y[int(y.shape[0]*split):,:]
            y_train = y[:int(y.shape[0]*split),:]

            x_train_s = np.zeros((x_train.shape[0],x_train.shape[1],x_train.shape[2]))
            x_test_s = np.zeros((x_test.shape[0],x_test.shape[1],x_test.shape[2]))
            # Scale each feature of X
            for i in range(X_toScale.shape[1]):
                scaler_x = MinMaxScaler(feature_range=(0, 1))
                scaler_y = MinMaxScaler(feature_range=(0, 1))

                scaler_x.fit(x_train[:,i,:])

                x_train_s[:,i,:] = scaler_x.transform(x_train[:,i,:])
                x_test_s[:,i,:] = scaler_x.transform(x_test[:,i,:])
                # Save the scalers
                with open(f'scaler{i}x.pkl', 'wb') as file:
                    pickle.dump(scaler_x, file)
                    print('Scaler saved')

            # Scale each feature of y
            scaler_y.fit(y_train)

            y_train_s = np.zeros((y_train.shape))
            y_test_s = np.zeros((y_test.shape))

            y_train_s = scaler_y.transform(y_train)
            y_test_s = scaler_y.transform(y_test)

            y_test_s = y_test_s.reshape(y_test_s.shape[0],y_test_s.shape[1],1)
            y_train_s = y_train_s.reshape(y_train_s.shape[0],y_train_s.shape[1],1)
            # Save the scaler
            with open('scalery.pkl', 'wb') as file:
                pickle.dump(scaler_y, file)
            print('Scaler saved')


            return x_train_s, y_train_s, x_test_s, y_test_s

        else:
            # Load the scalers and scale the provided dataset
            x_scaled = np.zeros((X_toScale.shape[0],X_toScale.shape[1],X_toScale.shape[2]))
            for i in range(X_toScale.shape[1]):
                with open(f'scaler{i}x.pkl', 'rb') as file:
                    scaler_x = pickle.load(file)
                    print('Scaler Loaded')
                    x_scaled[:,i,:] = scaler_x.transform(X_toScale[:,i,:])

            return x_scaled
    
    def trainModel(self, x_train, y_train, x_test, y_test, epc):
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
            epochs=epc, 
            batch_size=50, 
            validation_data=(x_test,y_test), 
            shuffle=False,
        )

        model.save('my_model.h5') 
        print('Model Saved')
        return model
    
    def runModel(self, X):
        print('Predicting')
        model = load_model('my_model.h5')
        results = model.predict(X)
        print('Descaling')
        #results = self.descaleData(results)
        
        return results




# Train
p = Preprocess()
data = p.loadData()

data

X = p.prepareData(data)

X.shape

X, y =  p.rollByLags(X)

mgp = MGP()
a,b,c,d = mgp.scaleData(X, y)

mgp.trainModel(
    a,b,c,d,
    20
)

plt.rcParams['figure.figsize'] = (10,7)
plt.plot(mgp.history.history['loss'], label='Train', linewidth=3)
plt.plot(mgp.history.history['val_loss'], label='Test', linewidth=3, linestyle='-.', color = 'r')
plt.legend()
plt.grid(linestyle = '--')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.xlim(0)

# Validation
p = Preprocess()
data = p.loadData()
X_val = p.prepareData(data)
y_val = X_val[:,:,-1]
mgp = MGP()
y_hat = mgp.runModel(
    mgp.scaleData(X_val)
)

y_hat_d = mgp.descaleData(y_hat)

LIM = 400
plt.rcParams['figure.figsize'] = (20,10)
plt.plot(y_val[:LIM,0], label='Observation')
plt.plot(y_hat_d[:LIM,0], label='Predictions', linestyle='-.', color = 'r')
plt.legend()
plt.grid(linestyle = '--')
plt.xlabel('Timesteps')
plt.ylabel('Daily Quantity')
plt.xlim(0)

# Evaluate Model
plt.scatter(y_val, y_hat_d, s=.5)
plt.plot(y_val,y_val, color='r')
plt.show()


