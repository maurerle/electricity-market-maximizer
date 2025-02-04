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
        # Resampling
        off = off.set_index(pd.DatetimeIndex(off.index))
        off = off.resample('D').mean()
        # Get OFF(t-1) as feature
        off['DLY_QTY'] = np.roll(off['DLY_QTY'], 1, axis=0)
        # Get OFF(t) as feature
        label = np.roll(off['DLY_QTY'], -1, axis=0)
        #label = np.roll(off['DLY_QTY'], -7, axis=0)
        # Discard the first values
        off = off[1:]
        label = label[1:]
        #off = off[7:]
        #label = label[7:]
        
        # Read MGP
        mgp = (
            pd
            .read_csv('mgp.csv', index_col='Timestamp')
            .iloc[1:]
        )
        mgp = mgp.set_index(pd.DatetimeIndex(mgp.index))
        mgp = mgp.resample('D').mean()
        mgp = mgp.iloc[1:-6]

        # Merge dataframes
        merged = pd.merge(
            mgp, 
            off, 
            left_index=True, 
            right_index=True
        )
        
        return merged[self.selectFeatures(merged)], label
    
    def selectFeatures(self, dataframe):
        dataset = dataframe.drop(columns=['DLY_QTY']).dropna()
        cols=[x for x in dataset.head()]

        corr = dataframe.corr()
        corr = corr.drop(columns=cols).dropna()
        corr = corr.where(abs(corr['DLY_QTY'])>self.corr_limit).dropna()
        features = [x for x in corr.index]
        
        return features
    
    def reshapeData(self, data, lags):
        data1 = np.zeros(shape=(data.shape[0],1,data.shape[1]), dtype=float)
        data1[:,0,:] = data
        data2 = np.zeros(shape=(data.shape[0],lags,data.shape[1]), dtype=float)
        for i in range(data.shape[0]-lags+1):
            for j in np.arange(0,lags):
                data2[i,j] = data1[j+i]
        return data2
    
    def rollByLags(self, X, lags):
        lag = -lags
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

    def trainModel(self, x_train, y_train, x_test, y_test, epc, unit1, unit2, drop, batch):
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
            batch_size=batch, 
            validation_data=(x_test,y_test), 
            shuffle=False,
        )

        model.save('my_model.h5') 

        return model

# # Model Training
# ## Single lag
# Not fully correct version of LSTM


# Train
p = Preprocess(.4)
data , label= p.loadData()
data

X = np.asarray(data)
y = label

# +
SPLIT = int(.7*X.shape[0])

# MANAGE X
x_train_u = X[:SPLIT]
x_test_u = X[SPLIT:]

# Scale Xtrain and Xtest
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_x.fit(x_train_u)
x_train = scaler_x.transform(x_train_u)
x_test = scaler_x.transform(x_test_u)
# Save the scaler
with open('scaler_x.pkl', 'wb') as file:
    pickle.dump(scaler_x, file)

# MANAGE y
y = y.reshape(-1, 1)
y_train_u = y[:SPLIT]
y_test_u = y[SPLIT:]

# Scale Xtrain and Xtest
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_y.fit(y_train_u)
y_train = scaler_y.transform(y_train_u)
y_test = scaler_y.transform(y_test_u)
# Save the scaler
with open('scaler_y.pkl', 'wb') as file:
    pickle.dump(scaler_y, file)
# -

x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
y_train = y_train.reshape(y_train.shape[0], 1, y_train.shape[1])
y_test = y_test.reshape(y_test.shape[0], 1, y_test.shape[1])

# +
model = Sequential()

model.add(
    LSTM(
        200, 
        input_shape=(x_train.shape[1], x_train.shape[2]), 
        return_sequences=True, 
        unroll=True
    )
)
model.add(Dropout(.2))

model.add(
    Bidirectional(
        LSTM(
            80, 
            input_shape=(x_train.shape[1], x_train.shape[2]),
            unroll=True,
            return_sequences=True, 
        )
    )
)
model.add(Dropout(.2))

model.add(TimeDistributed(Dense(1)))

opt = Adam(learning_rate=0.001)

model.compile(loss='mse', optimizer=opt , metrics = ['mae', 'mape'])
print(model.summary())

history = model.fit(
    x_train,
    y_train,
    epochs=100, 
    batch_size=20, 
    validation_data=(x_test,y_test), 
    shuffle=False,
)

model.save('my_model.h5')
# -

plt.plot(history.history['loss'], label='Train', linewidth=3)
plt.plot(history.history['val_loss'], label='Test', linewidth=3, linestyle='-.', color = 'r')
plt.legend()
plt.grid(linestyle = '--')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.xlim(0)





















# ## Timesteps version of LSTM
# Correct version. If lag is equal to 7, then   \[0 1 2 3 4 5 6\] -> \[7 8 9 10 11 12 13\]  
# So the important data is 6 whose target is 7

# Train
p = Preprocess(.4)
data , label= p.loadData()
label = label.reshape(-1, 1)

# Create X and y datasets
X = p.reshapeData(data, 7)
y = p.reshapeData(label, 7)
y = y.reshape(y.shape[0], y.shape[1])

# Scaling Data
mgp = MGP()
a,b,c,d = mgp.scaleData(.7, X, y)

# +
mgp.trainModel(a,b,c,d, 1000, 200, 80, .2, 20)

plt.rcParams['figure.figsize'] = (10,7)
plt.plot(mgp.history.history['loss'], label='Train', linewidth=3)
plt.plot(mgp.history.history['val_loss'], label='Test', linewidth=3, linestyle='-.', color = 'r')
plt.legend()
plt.grid(linestyle = '--')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.xlim(0)
# -
# # Validation


# Validation
p = Preprocess(.4)
data , label= p.loadData()
label = label.reshape(-1, 1)

X = p.reshapeData(data, 7)
y = p.reshapeData(label, 7)
y = y.reshape(y.shape[0], y.shape[1])

scaled = mgp.scaleData(.7, X)
model = load_model('my_model.h5')

LIMIT = scaled.shape[0]
y_hat = []
for i in range(LIMIT):
    y_hat.append(model.predict(scaled[i].reshape(1,scaled.shape[1], scaled.shape[2]))[0])

y_hat = np.asarray(y_hat)
y_hat_d = mgp.descaleData(y_hat)

plt.plot(y[600:650,-1], label='Observation', marker='o')
plt.plot(y_hat_d[600:650,-1], linestyle='-.', label='LSTM', marker='*', color='r')
plt.legend()
plt.grid(linestyle = '--')
plt.xlabel('Days')
plt.ylabel('Daily Average Quantity')
plt.xlim(0)

plt.scatter(y[:,0], y_hat_d[:,0], s=7)
plt.plot(y[:,0],y[:,0], color='r')
#plt.show()





















# ## Try to feed only one y

# Train
p = Preprocess(.4)
data , label= p.loadData()
label = label.reshape(-1, 1)

X = p.reshapeData(data, 7)
y = label
y = y.reshape(y.shape[0], y.shape[1])

# Scaling Data
mgp = MGP()
a,b,c,d = mgp.scaleData(.7, X, y)

temp_b = b.reshape(b.shape[0])
temp_d = d.reshape(d.shape[0])

# +
model = Sequential()
model.add(LSTM(100, unroll=True, return_sequences=True, input_shape=(a.shape[1], a.shape[2])))
model.add(Dropout(.2))
model.add(Bidirectional(LSTM(50), input_shape=(a.shape[1], a.shape[2])))
model.add(Dropout(.2))
model.add(Dense(1))
opt = Adam(learning_rate=0.0001)

model.compile(loss='mse', optimizer=opt , metrics = ['mae', 'mape'])
#print(model.summary())
history = model.fit(
    a,
    temp_b,
    epochs=1000, 
    batch_size=20, 
    validation_data=(c,temp_d), 
    shuffle=False,
)

# +
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(a.shape[1], a.shape[2])))
model.add(Dense(1))
opt = Adam(learning_rate=0.0001)

model.compile(loss='mse', optimizer=opt , metrics = ['mae', 'mape'])
#print(model.summary())
history = model.fit(
    a,
    temp_b,
    epochs=1000, 
    batch_size=1, 
    validation_data=(c,temp_d), 
    shuffle=False,
)
# -

plt.rcParams['figure.figsize'] = (10,7)
plt.plot(history.history['loss'], label='Train', linewidth=3)
plt.plot(history.history['val_loss'], label='Test', linewidth=3, linestyle='-.', color = 'r')
plt.legend()
plt.grid(linestyle = '--')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.xlim(0)




















# ## Old Version of LSTM
# Same version of hourly data, but daily resampled

X = p.reshapeData(data, 7)

X, y =  p.rollByLags(X, 7)
print(f"Scaled Dataset shape:{X.shape}:")
print(f"\tN samples: {X.shape[0]}")
print(f"\tN lags: {X.shape[1]}")
print(f"\tN features: {X.shape[2]}")

y.shape

# Scaling Data
mgp = MGP()
a,b,c,d = mgp.scaleData(.7, X, y)

# +
mgp.trainModel(a,b,c,d, 100, 200, 80, .2, 20)

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
data , label= p.loadData()
X_val = p.reshapeData(data, 7)

y_val = X_val[:,:,-1]
mgp = MGP()

scaled = mgp.scaleData(.7, X_val)
model = load_model('my_model.h5')

LIMIT = 800
y_hat = []
for i in range(LIMIT):
    y_hat.append(model.predict(scaled[i].reshape(1,scaled.shape[1], scaled.shape[2]))[0])

y_hat = np.asarray(y_hat)
y_hat_d = mgp.descaleData(y_hat)

y_val.shape

plt.plot(np.arange(5,805),y_val[:800,-1], label='Observation', linewidth='1.7')
plt.plot(np.arange(0,800),y_hat_d[:,0], linestyle='-.', label='Predictions', linewidth='1.7')
plt.legend()
plt.grid(linestyle = '--')
plt.xlabel('Timesteps')
plt.ylabel('Daily Quantity')
plt.xlim(0)

plt.scatter(y_val[24:LIMIT+24,0], y_hat_d[:,0], s=.5)
plt.plot(y_val[24:LIMIT+24,0],y_val[24:LIMIT+24,0], color='r')
plt.show()



















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
)
# Resampling
off = off.set_index(pd.DatetimeIndex(off.index))
off = off.resample('D').mean()
pd.set_option("display.max_rows", None)
off
# -

# Get OFF(t-1) as feature
off['DLY_QTY'] = np.roll(off['DLY_QTY'], 1, axis=0)
# Get OFF(t) as feature
label = np.roll(off['DLY_QTY'], -1, axis=0)
#label = np.roll(off['DLY_QTY'], -7, axis=0)
# Discard the first values
off = off[1:]
label = label[1:]
#off = off[7:]
#label = label[7:]
off

pd.DataFrame(label)







p = Preprocess(.4)
data , label= p.loadData()
label = label.reshape(-1, 1)
data

X = p.reshapeData(data, 7)
y = p.reshapeData(label, 7)
y = y.reshape(y.shape[0], y.shape[1])

pd.DataFrame(X[0])

pd.DataFrame(y[0])

# Scaling Data
mgp = MGP()
a,b,c,d = mgp.scaleData(.7, X, y)

pd.DataFrame(a[0])

pd.DataFrame(b[0])

plt.plot(a[100:150,1,-1])
plt.plot(b[100:150,0])


