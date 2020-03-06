import pandas as pd
import numpy as np 
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import pickle

TIMESTEPS = 7
WINDOW = 35
MIN_CORR = .1

class Preprocess():
    def __init__(self):
        pass

    def loadData(self):
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
        off = off.resample('D').sum()
        # Get OFF(t-1) as feature
        off['DLY_QTY'] = np.roll(off['DLY_QTY'], 1, axis=0)
        # Get OFF(t) as feature
        label = np.roll(off['DLY_QTY'], -1, axis=0)
        # Discard the first values
        off = off[1:]
        label = label[1:]
        
        # Read MGP
        mgp = (
            pd
            .read_csv('data/mgp.csv', index_col='Timestamp')
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
        corr = corr.where(abs(corr['DLY_QTY'])>MIN_CORR).dropna()
        features = [x for x in corr.index]
        
        return features
    
    def scaleData(self, size, X_toScale, *y):

        if len(y)>0:
            y = y[0]

            # Initialize the training and test items
            x_test = X_toScale[size,:].reshape(1,-1)
            x_train = X_toScale[:size,:]
            y_test = y[size].reshape(1,-1)
            y_train = y[:size]
            
            x_train_s = np.zeros((x_train.shape[0],x_train.shape[1]))
            x_test_s = np.zeros((x_test.shape))
            # Scale each feature of X
            scaler_x = MinMaxScaler(feature_range=(0, 1))
            scaler_y = MinMaxScaler(feature_range=(0, 1))
            
            scaler_x.fit(x_train)

            x_train_s = scaler_x.transform(x_train)
            x_test_s = scaler_x.transform(x_test)
            # Save the scalers
            with open(f'models/svrScalerx.pkl', 'wb') as file:
                pickle.dump(scaler_x, file)

            # Scale each feature of y
            scaler_y.fit(y_train)

            y_train_s = np.zeros((y_train.shape))

            y_train_s = scaler_y.transform(y_train)
            y_test_s = scaler_y.transform(y_test)

            # Save the scaler
            with open('models/svrScalery.pkl', 'wb') as file:
                pickle.dump(scaler_y, file)


            return x_train_s, y_train_s, x_test_s, y_test_s

        else:
            # Load the scalers and scale the provided dataset
            x_scaled = np.zeros((X_toScale.shape[0],X_toScale.shape[1]))
            with open(f'models/svrScalerx.pkl', 'rb') as file:
                scaler_x = pickle.load(file)
                x_scaled = scaler_x.transform(X_toScale)

            return x_scaled
        
    def descaleData(self, y_hat):
        with open('models/svrScalery.pkl', 'rb') as file:
            scaler_y = pickle.load(file)

        y_hat_d = scaler_y.inverse_transform(y_hat)

        return y_hat_d

# ## 1Day Validation
print('Performing 1 day validation')
p = Preprocess()
original_data, _= p.loadData()

N=1
y = []
y_hat = []
for j in range(original_data.shape[0]):
    # Emulate the next day
    temp_data = original_data.copy()
    try:
        for i in range(N):
            pred = []
            # Load the last available sample
            X = temp_data.values[j+i:i+j+WINDOW+1].reshape(WINDOW+1,-1)
            target = temp_data.values[i+j+1:i+j+WINDOW+2, -1].reshape(WINDOW+1,-1)
            # Scale Data
            x_train, y_train, x_test, y_test = p.scaleData(WINDOW, X, target)
            model = SVR(kernel='rbf',gamma='auto',C=1.0,epsilon=0.1)
            model.fit(x_train, y_train[:,0])
            # Predict the next missing sample of Public Offers
            pred.append(model.predict(x_test))
            missing_off = p.descaleData(np.asarray(pred))[0,0]
            # If i is 7, the missing week has been
            # forecasted, so the original sample is put in y, whereas
            # the forecasted one is the target one (t+1)
            if i == N-1:
                y.append(p.descaleData(np.asarray(y_test))[0,0])
                y_hat.append(missing_off)
            # Replace the real data retrieved for training purpose with
            # the forecasted sample.
            temp_data.iloc[i+j+1]['DLY_QTY'] = missing_off
    except:
        pass

to_save = {
    'y':y,
    'y_hat':y_hat
}
pd.DataFrame(to_save).to_csv('data/svr1days.csv')

# ## 8Days Validation
print('Performing 8 day validation')
p = Preprocess()
original_data, _= p.loadData()

N=8
y = []
y_hat = []
for j in range(original_data.shape[0]):
    # Emulate the next day
    temp_data = original_data.copy()
    try:
        for i in range(N):
            pred = []
            # Load the last available sample
            X = temp_data.values[j+i:i+j+WINDOW+1].reshape(WINDOW+1,-1)
            target = temp_data.values[i+j+1:i+j+WINDOW+2, -1].reshape(WINDOW+1,-1)
            # Scale Data
            x_train, y_train, x_test, y_test = p.scaleData(WINDOW, X, target)
            model = SVR(kernel='rbf',gamma='auto',C=1.0,epsilon=0.1)
            model.fit(x_train, y_train[:,0])
            # Predict the next missing sample of Public Offers
            pred.append(model.predict(x_test))
            missing_off = p.descaleData(np.asarray(pred))[0,0]
            # If i is 7, the missing week has been
            # forecasted, so the original sample is put in y, whereas
            # the forecasted one is the target one (t+1)
            if i == N-1:
                y.append(p.descaleData(np.asarray(y_test))[0,0])
                y_hat.append(missing_off)
            # Replace the real data retrieved for training purpose with
            # the forecasted sample.
            temp_data.iloc[i+j+1]['DLY_QTY'] = missing_off
    except:
        pass

to_save = {
    'y':y,
    'y_hat':y_hat
}
pd.DataFrame(to_save).to_csv('data/svr8days.csv')


