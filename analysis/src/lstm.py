from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, Bidirectional, TimeDistributed
from keras.optimizers import Adam
import pandas as pd 
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import rcParams
import pickle

SPLIT = .7
TIMESTEPS = 7
MIN_CORR = .5


class Preprocess():
    """Data preprocessor class for the LSTM model.
    
    Parameters
    ----------
    corr_limit : Float
        Minimum level of correlation among features
    
    Attributes
    -------
    corr_limit : Float
        Minimum level of correlation among features
    
    Methods
    -------
    loadData()
        Data loader
    selectFeatures(dataframe, target)
        Selector of the features according to their correlation
    reshapeData(data)
        Data reshaper
    descaleData(y_hat)
        Data descaler
    scaleData(X_toScale, *y)
        Data scaler
    """
    def __init__(self, corr_limit):
        self.corr_limit = corr_limit

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
            .read_csv('../data/bid.csv', index_col='Unnamed: 0')
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
        # Get OFF(t) as feature
        label = np.roll(off['DLY_QTY'], -1, axis=0)
        # Discard the first values
        off = off[:-1]
        label = label[:-1]
        
        # Read MGP
        mgp = (
            pd
            .read_csv('../data/mgp.csv', index_col='Timestamp')
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
        corr = corr.where(abs(corr['Target'])>self.corr_limit).dropna()
        features = [x for x in corr.index if x != 'Target']
        
        return features
    
    def reshapeData(self, data):
        """Reshape a 2-d Dataframe to 3-d.
        
        Parameters
        ----------
        data : pandas.DataFrame
            2-d Data
        
        Returns
        -------
        pandas.DataFrame
            3-d Data
        """
        data1 = np.zeros(shape=(data.shape[0],1,data.shape[1]), dtype=float)
        data1[:,0,:] = data
        data2 = np.zeros(shape=(data.shape[0],TIMESTEPS,data.shape[1]), dtype=float)
        for i in range(data.shape[0]-TIMESTEPS+1):
            for j in np.arange(0,TIMESTEPS):
                data2[i,j] = data1[j+i]
        return data2
    
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
        with open('../models/lstmScalery.pkl', 'rb') as file:
            scaler_y = pickle.load(file)

        y_hat_r = y_hat.reshape(y_hat.shape[0],y_hat.shape[1])
        y_hat_d = scaler_y.inverse_transform(y_hat_r)

        return y_hat_d
    
    def scaleData(self, X_toScale, *y):
        """Scale the data to [0, 1] range.
        
        Parameters
        ----------
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
                with open(f'../models/lstmScaler{i}x.pkl', 'wb') as file:
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
            with open('../models/lstmScalery.pkl', 'wb') as file:
                pickle.dump(scaler_y, file)


            return x_train_s, y_train_s, x_test_s, y_test_s

        else:
            # Load the scalers and scale the provided dataset
            x_scaled = np.zeros((X_toScale.shape[0],X_toScale.shape[1],X_toScale.shape[2]))
            for i in range(X_toScale.shape[1]):
                with open(f'../models/lstmScaler{i}x.pkl', 'rb') as file:
                    scaler_x = pickle.load(file)
                    x_scaled[:,i,:] = scaler_x.transform(X_toScale[:,i,:])

            return x_scaled

class Correlation():
    """Class to perform correlation analysis for the LSTM model.
    
    Parameters
    ----------
    corr_limit : Float
        Minimum level of correlation among features
    
    Attributes
    -------
    corr_limit : Float
        Minimum level of correlation among features
    
    Methods
    -------
    loadData()
        Data loader
    selectFeatures(dataframe, target)
        Selector of the features according to their correlation
    """
    def __init__(self, corr_limit):
        self.corr_limit = corr_limit

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
            .read_csv('../data/bid.csv', index_col='Unnamed: 0')
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
            .read_csv('../data/mgp.csv', index_col='Timestamp')
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
        
        return merged, label
    
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
        pandas.DataFrame
            Dataframe of the selected features
        """
        dataset = dataframe.copy()
        cols=[x for x in dataframe.head()]
        dataset['Target'] = target

        corr = dataset.corr()
        corr = corr.drop(columns=cols).dropna()
        corr = corr.where(abs(corr['Target'])>self.corr_limit).dropna()
        features = [x for x in corr.index if x != 'Target']
    
        
        return corr