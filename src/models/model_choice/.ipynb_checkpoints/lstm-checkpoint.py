# +
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Bidirectional, TimeDistributed
from keras.optimizers import Adam
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
#from src.common.mongo import MongoDB as Mongo
from sklearn.preprocessing import MinMaxScaler
from matplotlib import rcParams
import pickle

rcParams.update({'font.size': 18})
# -

SPLIT = .7
TIMESTEPS = 7
MIN_CORR = .5


class Preprocess():
    def __init__(self, corr_limit):
        self.corr_limit = corr_limit

    def loadData(self):
        #mng = Mongo(logger, USER, PASSWD)
        
        #bid, off = mng.operatorAggregate(start, operator)
        #mgp = mng.mgpAggregate(start)
        
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
        )
        # Resampling
        off = off.set_index(pd.DatetimeIndex(off.index))
        off = off.resample('D').mean()
        # Get OFF(t) as feature
        label = np.roll(off['DLY_QTY'], -1, axis=0)
        # Discard the first values
        off = off[:-1]
        label = label[:-1]
        
        # Read MGP
        mgp = (
            pd
            .read_csv('data/mgp.csv', index_col='Timestamp')
            .iloc[1:]
        )
        mgp = mgp.set_index(pd.DatetimeIndex(mgp.index))
        mgp = mgp.resample('D').mean()
        mgp = mgp.iloc[:-(TIMESTEPS)]

        # Merge dataframes
        merged = pd.merge(
            mgp, 
            off, 
            left_index=True, 
            right_index=True
        )
        
        return merged[self.selectFeatures(merged, label)], label
    
    def selectFeatures(self, dataframe, target):
        dataset = dataframe.copy()
        cols=[x for x in dataframe.head()]
        dataset['Target'] = target

        corr = dataset.corr()
        corr = corr.drop(columns=cols).dropna()
        corr = corr.where(abs(corr['Target'])>self.corr_limit).dropna()
        features = [x for x in corr.index if x != 'Target']
        
        plt.plot(corr)
        
        return features
    
    def reshapeData(self, data):
        data1 = np.zeros(shape=(data.shape[0],1,data.shape[1]), dtype=float)
        data1[:,0,:] = data
        data2 = np.zeros(shape=(data.shape[0],TIMESTEPS,data.shape[1]), dtype=float)
        for i in range(data.shape[0]-TIMESTEPS+1):
            for j in np.arange(0,TIMESTEPS):
                data2[i,j] = data1[j+i]
        return data2
    
    def descaleData(self, y_hat):
        with open('models/lstmScalery.pkl', 'rb') as file:
            scaler_y = pickle.load(file)

        y_hat_r = y_hat.reshape(y_hat.shape[0],y_hat.shape[1])
        y_hat_d = scaler_y.inverse_transform(y_hat_r)

        return y_hat_d
    
    def scaleData(self, X_toScale, *y):

        if len(y)>0:
            y = y[0]

            # Initialize the training and test items
            x_test = X_toScale[int(X_toScale.shape[0]*SPLIT):,:,:]
            x_train = X_toScale[:int(X_toScale.shape[0]*SPLIT),:,:]
            
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
                with open(f'models/lstmScaler{i}x.pkl', 'wb') as file:
                    pickle.dump(scaler_x, file)

            
            y_test = y[int(y.shape[0]*SPLIT):,:]
            y_train = y[:int(y.shape[0]*SPLIT),:]
            
            # Scale each feature of y
            scaler_y.fit(y_train)

            y_train_s = np.zeros((y_train.shape))
            y_test_s = np.zeros((y_test.shape))

            y_train_s = scaler_y.transform(y_train)
            y_test_s = scaler_y.transform(y_test)

            y_test_s = y_test_s.reshape(y_test_s.shape[0],y_test_s.shape[1],1)
            y_train_s = y_train_s.reshape(y_train_s.shape[0],y_train_s.shape[1],1)
            # Save the scaler
            with open('models/lstmScalery.pkl', 'wb') as file:
                pickle.dump(scaler_y, file)


            return x_train_s, y_train_s, x_test_s, y_test_s

        else:
            # Load the scalers and scale the provided dataset
            x_scaled = np.zeros((X_toScale.shape[0],X_toScale.shape[1],X_toScale.shape[2]))
            for i in range(X_toScale.shape[1]):
                with open(f'models/lstmScaler{i}x.pkl', 'rb') as file:
                    scaler_x = pickle.load(file)
                    x_scaled[:,i,:] = scaler_x.transform(X_toScale[:,i,:])

            return x_scaled


# # Model Training


# Train
p = Preprocess(MIN_CORR)
data , label= p.loadData()
label = label.reshape(-1, 1)

for i in data.head():
    print(i)

# Create X and y datasets
X = p.reshapeData(data)
y = p.reshapeData(label)
y = y.reshape(y.shape[0], y.shape[1])

# Scaling Data
x_train, y_train, x_test, y_test = p.scaleData(X, y)

# +
# LSTM model
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

opt = Adam(learning_rate=0.0001)

model.compile(loss='mse', optimizer=opt , metrics = ['mae', 'mape'])

history = model.fit(
    x_train,
    y_train,
    epochs=10, 
    batch_size=20, 
    validation_data=(x_test,y_test), 
    shuffle=False,
)

model.save('models/lstm.h5') 
# -

plt.plot(history.history['loss'], label='Training', linewidth=2.5)
plt.plot(history.history['val_loss'], label='Test', linewidth=2.5, linestyle='-.', color = 'r')
plt.legend()
plt.grid(linestyle = '--')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.savefig('fig/loss.png', transparent=True)

# # Next Day Validation


# Validation
p = Preprocess(MIN_CORR)
data , label= p.loadData()
label = label.reshape(-1, 1)

X = p.reshapeData(data)
y = p.reshapeData(label)
y = y.reshape(y.shape[0], y.shape[1])

scaled = p.scaleData(X)
model = load_model('models/lstm.h5')

y_hat = []
for i in range(scaled.shape[0]):
    y_hat.append(model.predict(scaled[i].reshape(1,scaled.shape[1], scaled.shape[2]))[0])

y_hat = np.asarray(y_hat)
y_hat = p.descaleData(y_hat)

to_save = {
    'y':y[:,-1],
    'y_hat':y_hat[:,-1]
}
pd.DataFrame(to_save).to_csv('data/lstm1days.csv')

# # 8th Day Validation

# Validation
p = Preprocess(MIN_CORR)
original_data ,_= p.loadData()

y_hat = []
y = []
model = load_model('models/lstm.h5')
for j in range(original_data.shape[0]):
    # Emulate the next day
    temp_data = original_data.copy()
    try:
        for i in range(8):
            pred = []
            # Load TIMESTEPS samples
            data = temp_data.iloc[i+j:i+j+TIMESTEPS].copy()
            # Create one single sample of 3D data
            X = p.reshapeData(data)
            X = X[0,:,:]
            X = X.reshape(1,X.shape[0],X.shape[1])
            # Scale Data
            scaled = p.scaleData(X)
            # Predict the next missing sample of Public Offers
            pred.append(model.predict(scaled.reshape(1,scaled.shape[1], scaled.shape[2]))[0])
            missing_off = p.descaleData(np.asarray(pred))[0,-1]
            # If i is 7, the missing week has been
            # forecasted, so the original sample is put in y, whereas
            # the forecasted one is the target one (t+1)
            if i == 7:
                y.append(original_data.iloc[i+j+TIMESTEPS]['DLY_QTY'])
                y_hat.append(missing_off)
            # Replace the real data retrieved for training purpose with
            # the forecasted sample.
            temp_data.iloc[i+j+TIMESTEPS]['DLY_QTY'] = missing_off
    except:
        pass

to_save = {
    'y':y,
    'y_hat':y_hat
}
pd.DataFrame(to_save).to_csv('data/lstm8days.csv')


