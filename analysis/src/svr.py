from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sys
sys.path.append("..")

MIN_CORR = .1
N_WINDOW = 62

class Svr():
    """Class containing the methods for the SVR model.
    
    Methods
    -------
    loadData()
        Data loader
    selectFeatures(dataframe, target)
        Selector of the features according to their correlation
    scaleData(size, X_toScale, *y)
        Data scaler
    descaleData(y_hat)
        Data descaler
    """
    def __init__(self):
        pass

    def loadData(self):
        """Load the data from a .csv file and put them
        into a Pandas Dataframe.
        
        Returns
        -------
        pandas.DataFrame, numpy.ndarray
            Dataframe containing the data and the relative array of labels
        """
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
        # Daily Resampling
        off = off.set_index(pd.DatetimeIndex(off.index))
        off = off.resample('D').sum()
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
        mgp = mgp.iloc[:-7]

        # Merge dataframes
        merged = pd.merge(
            mgp, 
            off, 
            left_index=True, 
            right_index=True
        )
        
        return merged[self.selectFeatures(merged, label)], label

    def selectFeatures(self, dataframe, target):
        """Select the features of interest according to 
        the wanted minimum correlation among them.
        
        Parameters
        ----------
        dataframe : pandas.DataFrame
            Data
        target : numpy.ndarray
            The target variables
        
        Returns
        -------
        List
            The list of selected features
        """
        dataset = dataframe.copy()
        cols=[x for x in dataframe.head()]
        dataset['Target'] = target

        corr = dataset.corr()
        corr = corr.drop(columns=cols).dropna()
        corr = corr.where(abs(corr['Target'])>MIN_CORR).dropna()
        features = [x for x in corr.index if x != 'Target']
        
        return features
    
    def scaleData(self, size, X_toScale, *y):
        """Scale the data to [0, 1] range.
        
        Parameters
        ----------
        size : int
            Size of the training set
        X_toScale : numpy.ndarray
            Data to be scaled
        
        Returns
        -------
        numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray
            Scaled arrays of train and test set
        """
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
        """Descale the data from [0, 1] to original.
        
        Parameters
        ----------
        y_hat : numpy.ndarray
            Array to be descaled
        
        Returns
        -------
        numpy.ndarray
            Descaled array
        """
        with open('models/svrScalery.pkl', 'rb') as file:
            scaler_y = pickle.load(file)

        y_hat_d = scaler_y.inverse_transform(y_hat)

        return y_hat_d