import pandas as pd
from src.common.dataProcessing import DataProcessing
import logging
import logging.config
import numpy as np 
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Bidirectional, BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

logging.config.fileConfig('src/common/logging.conf')
logger = logging.getLogger(__name__)


def scaling(X, *y):
    global scaler
    
    scaler = StandardScaler()
    
    if len(y)>0:
        y = y[0]
        data = np.concatenate((X,y),axis=1)
    
        data = pd.DataFrame(scaler.fit_transform(data))

        y = data[data.shape[1]-1].to_numpy()
        X = data[range(data.shape[1]-1)].to_numpy() \
                                        .reshape(X.shape[0], 1, X.shape[1])

        return X, y
    else:
        y = np.ones((X.shape[0], 1), dtype=float)
        data = np.concatenate((X,y),axis=1)
    
        data = pd.DataFrame(scaler.fit_transform(data))

        y = data[data.shape[1]-1].to_numpy()
        X = data[range(data.shape[1]-1)].to_numpy() \
                                        .reshape(X.shape[0], 1, X.shape[1])

        return X
    

def descaling(x_test, results, *y_test):
    # Reshaping
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[2])
    results = results.reshape(len(results), 1)
    print(x_test.shape)
    print(results.shape)
    # Descaling predictions
    df_hat = np.concatenate((x_test, results), axis=1)
    print(df_hat.shape)
    df_hat = scaler.inverse_transform(df_hat)
    y_hat = df_hat[:,df_hat.shape[1]-1]

    if len(y_test)>0:
        # Reshaping targets
        y_test = y_test[0]
        y_test = y_test.reshape(len(y_test), 1)
        
        # Descaling targets
        df_target = np.concatenate((x_test, y_test),axis=1)
        df_target = scaler.inverse_transform(df_target)
        y = df_target[:,df_hat.shape[1]-1]

        return y, y_hat 
    else:
        return y_hat




class MGP():
    def __init__ (self, user, passwd):
        self.user = user
        self.passwd = passwd
        self.b_qty_model = None
        self.lag = 1

    def createSet(self, start, operator):
        mongo = DataProcessing(logger, self.user, self.passwd)
        
        dataset = mongo.merge(
            mongo.mgpAggregate(start),
            mongo.operatorAggregate(operator, 'OffertePubbliche')
        )
        
        
        return dataset

    def prepareData(self, data, target=True):
        data = data.drop(columns=['TYPE', 'Unnamed: 0'])
        
        if target:
            X = data.iloc[:len(data.index)-self.lag,].to_numpy(dtype=float)
            X = PCA(n_components=70).fit_transform(X)
            
            new_data = data.shift(periods=-self.lag, fill_value=0)
            y = new_data['DLY_QTY'].iloc[:len(new_data.index)-self.lag,] \
                                   .to_numpy(dtype=float)
            new_data = new_data.drop(
                columns=['DLY_QTY','DLY_PRICE','DLY_AWD_QTY','DLY_AWD_PRICE']
            )
            y = y.reshape(len(y), 1)

            X, y = scaling(X, y)
        
            return X, y
        
        else:
            X = data.to_numpy(dtype=float)
            X = PCA(n_components=70).fit_transform(X)
            X = scaling(X)

            return X

    def train(self, data):
        X, y = self.prepareData(data, target=True)
        x_train, x_test, y_train, y_test = train_test_split(
            X,
            y, 
            test_size=.2, 
            random_state = 4
        )

        model = Sequential()
        
        model.add(
            LSTM(
                80, 
                input_shape=(X.shape[1], X.shape[2]), 
                return_sequences=True, 
                unroll=True
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(
            Bidirectional(
                LSTM(
                    50, 
                    input_shape=(X.shape[1], X.shape[2])
                )
            )
        )
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        
        model.add(Dense(40))
        model.add(Dropout(0.1))
        
        model.add(Dense(1))
        
        model.compile(loss='mse', optimizer='adam')
        print(model.summary())

        history = model.fit(
            x_train,
            y_train,
            epochs=50, 
            batch_size=50, 
            validation_data=(x_test,y_test), 
            shuffle=False
        )

        results = model.predict(x_test)

        y_test, y_hat_test = descaling(x_test, results, y_test)

        # Evaluate model
        rmse = sqrt(mean_squared_error(y_test, y_hat_test))
        r2 = r2_score(y_test, y_hat_test)
        mae = mean_absolute_error(y_test, y_hat_test)
        mape = np.mean(np.abs((y_test - y_hat_test) / y_test)) * 100

        print(f'RMSE Test:  {rmse}')
        print(f'MAE Test:  {mae}')
        print(f'MAPE: {mape}%')
        print(f'R2 Test:  {r2}')

        # Test Comparison
        plt.scatter(np.arange(y_hat_test.shape[0]), y_hat_test, s=1, color='r')
        plt.plot(y_test, linewidth=.5)
        plt.show()

        # Regression Line
        plt.xlabel('y')
        plt.ylabel('y_hat')
        plt.scatter(y_test,y_hat_test, s=1, label = 'y vs. y_hat')
        plt.plot(y_test, y_test, 'r', label = 'y = y')
        plt.legend()
        plt.show()

        # Loss Function
        plt.ylabel('MSE')
        plt.xlabel("Epochs")
        plt.plot(history.history['loss'], label = 'Test')
        plt.plot(history.history['val_loss'], label = 'Train')
        plt.legend()
        plt.show()


        model.save('models/my_model.h5') 
        
        return model


    def predict(self, data, *model):
        if len(model)>0:
            model = model[0]
        else:
            model = load_model('models/my_model.h5')
        
        X = self.prepareData(data, target=False)
        results = model.predict(X)

        predictions = descaling(X, results)

        return predictions

class MI():
    def __init__ (self):
        pass

class MSD():
    def __init__ (self):
        pass


    