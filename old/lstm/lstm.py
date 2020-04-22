# +
import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    from keras.models import Sequential, load_model
    from keras.layers import Dense, LSTM, Dropout, Bidirectional, BatchNormalization, Conv1D, MaxPooling1D, Flatten, RepeatVector, TimeDistributed
    from keras.optimizers import Adam
    
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
import pickle
import logging
import logging.config

plt.rcParams['figure.figsize'] = (10,7)


# -

class Preprocess():
    def __init__(self, corr_limit):
        self.X = []
        self.mat_list = []
        self.cnt = 1
        self.prev_hour = 0
        self.corr_limit = corr_limit

    def loadData(self):
        #mng = Mongo(logger, USER, PASSWD)
        
        #bid, off = mng.operatorAggregate(start, operator)
        #mgp = mng.mgpAggregate(start)
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

        return merged[self.selectFeatures(merged)]
    
    def selectFeatures(self, dataframe):
        dataset = dataframe.drop(columns=['DLY_QTY']).dropna()
        cols=[x for x in dataset.head()]

        corr = dataframe.corr()
        corr = corr.drop(columns=cols).dropna()
        corr = corr.where(abs(corr['DLY_QTY'])>self.corr_limit).dropna()
        features = [x for x in corr.index]
        
        return features
    
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

        return np.asarray(self.X)
    
    def rollByLags(self, X):
        lag = -24
        y = np.roll(X, lag, axis=0)[:,:,-1]
        X, y = X[:lag,:,:], y[:lag,:]

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
    
    def scaleData(self, split, X_toScale, *y):

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
                    #print('Scaler saved')

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
            #print('Scalers saved')


            return x_train_s, y_train_s, x_test_s, y_test_s

        else:
            # Load the scalers and scale the provided dataset
            x_scaled = np.zeros((X_toScale.shape[0],X_toScale.shape[1],X_toScale.shape[2]))
            for i in range(X_toScale.shape[1]):
                with open(f'scaler{i}x.pkl', 'rb') as file:
                    scaler_x = pickle.load(file)
                    #print('Scaler Loaded')
                    x_scaled[:,i,:] = scaler_x.transform(X_toScale[:,i,:])

            return x_scaled

    def trainModel(self, x_train, y_train, x_test, y_test, epc, unit1, unit2, drop):
        model = Sequential()

        model.add(
            LSTM(
                unit1, 
                input_shape=(x_train.shape[1], x_train.shape[2]), 
                return_sequences=True, 
                unroll=True
            )
        )
        model.add(Dropout(drop))

        model.add(
            Bidirectional(
                LSTM(
                    unit2, 
                    input_shape=(x_train.shape[1], x_train.shape[2]),
                    unroll=True,
                    return_sequences=True, 
                )
            )
        )
        model.add(Dropout(drop))

        model.add(TimeDistributed(Dense(1)))
        
        opt = Adam(learning_rate=0.0001)
        
        model.compile(loss='mse', optimizer=opt , metrics = ['mae', 'mape'])
        print(model.summary())
                
        self.history = model.fit(
            x_train,
            y_train,
            epochs=epc, 
            batch_size=10, 
            validation_data=(x_test,y_test), 
            shuffle=False,
        )

        model.save('my_model.h5') 

        return model

# # Model Training


# Train
p = Preprocess(.4)
data = p.loadData()
X = p.prepareData(data)
X, y =  p.rollByLags(X)
print(f"Scaled Dataset shape:{X.shape}:")
print(f"\tN samples: {X.shape[0]}")
print(f"\tN lags: {X.shape[1]}")
print(f"\tN features: {X.shape[2]}")

# Scaling Data
mgp = MGP()
a,b,c,d = mgp.scaleData(.8, X, y)

# +
mgp.trainModel(a,b,c,d, 80, 300, 80, .2)

plt.rcParams['figure.figsize'] = (10,7)
plt.plot(mgp.history.history['loss'], label='Train', linewidth=3)
plt.plot(mgp.history.history['val_loss'], label='Test', linewidth=3, linestyle='-.', color = 'r')
plt.legend()
plt.grid(linestyle = '--')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.xlim(0)
# -

temp = pd.DataFrame(mgp.history.history)
temp.to_csv('loss.dat')

# # Validation


# Validation
p = Preprocess(.4)
data = p.loadData()
X_val = p.prepareData(data)

y_val = X_val[:,:,-1]
mgp = MGP()

scaled = mgp.scaleData(.7, X_val)
model = load_model('my_model.h5')

LIMIT = 10000
y_hat = []
for i in range(LIMIT):
    y_hat.append(model.predict(scaled[i].reshape(1,scaled.shape[1], scaled.shape[2]))[0])

y_hat = np.asarray(y_hat)
y_hat_d = mgp.descaleData(y_hat)

y_val.shape

plt.plot(y_hat_d[200:500,0], linestyle='-.', label='Predictions', linewidth='1.7')
plt.plot(y_val[224:524,0], label='Observation', linewidth='1.7')
plt.legend()
plt.grid(linestyle = '--')
plt.xlabel('Timesteps')
plt.ylabel('Daily Quantity')
plt.xlim(0)

plt.scatter(y_val[24:LIMIT+24,0], y_hat_d[:,0], s=.5)
plt.plot(y_val[24:LIMIT+24,0],y_val[24:LIMIT+24,0], color='r')
plt.show()


