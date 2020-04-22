import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil import relativedelta
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.graphics.tsaplots as tplot
from statsmodels.tsa.arima_model import ARIMA
from influxdb import InfluxDBClient
from sklearn.metrics import mean_squared_error
import json
import warnings


with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

H = 1


class Arima():
    """Class with the methods of the ARIMA model.
    
    Parameters
    ----------
    op : str
        The operator of interest
    today : datetime.datetime
        The current day
    
    Attributes
    -------
    client : influxdb.InfluxDBClient
        InfluxDB Client
    op : str
        The operator of interest
    TODAY : datetime.datetime
        The current day
    
    Methods
    -------
    manageIndexTest(data)
        Manager of the test set index
    getDataTest(market)
        Getter of the test data
    runTest(data, label)
        Runner of the model
    predict()
        Make the prediction
    validationDem(market)
        Method for the validation of the demand
    validationSup(market)
        Method for the validation of the supply
    """
    def __init__(self, op, today):
        self.client = InfluxDBClient(
            'localhost', 
            8086, 
            'root', 
            'root', 
            'PublicBids'
        )
        self.op = op
        self.TODAY = today
        self.START = self.TODAY - relativedelta.relativedelta(days=60)
        self.TOMORROW = self.TODAY + relativedelta.relativedelta(days=1)

    def manageIndexTest(self, data):
        """Method to manage and adapt the indices of the dataframe.
        
        Parameters
        ----------
        data : pandas.Dataframe
            Initial Dataframe
        
        Returns
        -------
        pandas.Dataframe
            Dataframe with the converted indices
        """
        temp_date = pd.to_datetime(data['time'])
        data = (
            data
            .set_index(temp_date)
            .drop(columns=['time', 'OPS'])
        )
        idx = pd.date_range(self.START + relativedelta.relativedelta(days=1), self.TODAY)
        data.index = pd.DatetimeIndex(data.index.date)
        data = (data.reindex(idx, fill_value=0))
        
        return data

    def getDataTest(self, market):
        """Get the test set making queries to InfluxDB 
        
        Parameters
        ----------
        market : str
            The market of interest
        
        Returns
        -------
        pandas.Dataframe, pandas.Dataframe
            The Dataframes of demand and supply
        """
        res = (
            self.client
            .query(
                f"SELECT * FROM demand{market} WHERE op='{self.op}' AND time >= '{self.START}'"
            )
            .raw
        )
        try:
            dem =(
                pd
                .DataFrame(
                    res['series'][0]['values'], 
                    columns = ['time', 'P', 'Q', 'OPS']
                )
            )
            dem = self.manageIndexTest(dem)
        except:
            dem = -1
        # Get the supply data from InfluxDB
        res = (
            self.client
            .query(
                f"SELECT * FROM supply{market} WHERE op='{self.op}' AND time >= '{self.START}'"
            )
            .raw
        )
        try:
            sup =(
                pd
                .DataFrame(
                    res['series'][0]['values'], 
                    columns = ['time', 'P', 'Q', 'OPS']
                )
            )
            sup = self.manageIndexTest(sup)
        except:
            sup = -1

        return dem, sup


    def runTest(self, data, label):
        """Method which runs the ARIMA model
        
        Parameters
        ----------
        data : pandas.DataFrame
            Input data for the ARIMA model
        label : str
            A label
        
        Returns
        -------
        numpy.ndarray
            Array of out of sample forecasts
        """
        X = data.values.astype('float32')
        
        model = ARIMA(
            X, 
            order=(0,1,0),
        )
        model_fit = model.fit(maxiter=200, disp=0, method='css')
        y = model_fit.forecast(H)[0][-1]
        
        if y < 0:
            y = 0
        return y 

    def predict(self):
        """Make the predictions and call the validation methods to validate
        supply and demand data.
        
        Returns
        -------
        List, List
            Prediction and validations for supply and demand data
        """
        pred = []
        val = []
        for m in ['MGP', 'MI', 'MSD']:
            # Get the demand and supply curves
            d, s = self.getDataTest(m)
            # If the operator is not active in the market, getData 
            # returns -1, so the same -1 is appended to theprediction 
            # and it is managed later by the genetic algorithm
            if isinstance(s, int):
                pass
            else:
                pred.append(self.runTest(s.P, f'{m}pO'))
                pred.append(self.runTest(s.Q, f'{m}qO'))

                s_val_p, s_val_q = self.validationSup(m)
                val.append(s_val_p[0])
                val.append(s_val_q[0])
            # If the operator is not active in the market, getData 
            # returns -1, so the same -1 is appended to the prediction 
            # and it is managed later by the genetic algorithm            
            
            if isinstance(d, int):
                pass
            else:
                pred.append(self.runTest(d.P, f'{m}pD'))
                pred.append(self.runTest(d.Q, f'{m}qD'))

                d_val_p, d_val_q = self.validationDem(m)
                val.append(d_val_p[0])
                val.append(d_val_q[0])
        
        #return np.asarray(pred)
        return pred, val

    def validationDem(self, market):
        """Validate demand data.
        
        Parameters
        ----------
        market : str
            The market of interest
        
        Returns
        -------
        numpy.ndarray, numpy.ndarray
            Prices and quantities arrays
        """
        # Get the demand data from InfluxDB
        res = (
            self.client
            .query(
                f"SELECT * FROM demand{market} WHERE op='{self.op}' AND time = '{self.TOMORROW}'"
            )
            .raw
        )
        
        try:
            dem =(
                pd
                .DataFrame(
                    res['series'][0]['values'], 
                    columns = ['time', 'P', 'Q', 'OPS']
                )
            )

            return dem.P.values, dem.Q.values
        except:
            return [0.0], [0.0]
    
    def validationSup(self, market):
        """Validate supply data.
        
        Parameters
        ----------
        market : str
            The market of interest
        
        Returns
        -------
        numpy.ndarray, numpy.ndarray
            Prices and quantities arrays
        """
        # Get the supply data from InfluxDB
        res = (
            self.client
            .query(
                f"SELECT * FROM supply{market} WHERE op='{self.op}' AND time = '{self.TOMORROW}'"
            )
            .raw
        )
        try:
            sup =(
                pd
                .DataFrame(
                    res['series'][0]['values'], 
                    columns = ['time', 'P', 'Q', 'OPS']
                )
            )
            
            return sup.P.values, sup.Q.values
        except:
            return [0.0], [0.0]



class ArimaV2():
    """A second version of ARIMA model.
    
    Parameters
    ----------
    op : str
        The operator of interest
    today : datetime.datetime
        The current day
    
    Attributes
    -------
    client : influxdb.InfluxDBClient
        InfluxDB Client
    op : str
        The operator of interest
    TODAY : datetime.datetime
        The current day
    
    Methods
    -------
    manageIndexTest(data)
        Manager of the test set index
    getDataTest(market)
        Getter of the test data
    runTest(data, label)
        Runner of the model
    predict()
        Make the prediction
    """
    def __init__(self, op, today):
        self.client = InfluxDBClient(
            'localhost', 
            8086, 
            'root', 
            'root', 
            'PublicBids'
        )
        self.op = op
        self.TODAY = today
        self.START = self.TODAY - relativedelta.relativedelta(days=60)
        self.TOMORROW = self.TODAY + relativedelta.relativedelta(days=1)

    def manageIndexTest(self, data):
        """Method to manage and adapt the indices of the dataframe.
        
        Parameters
        ----------
        data : pandas.Dataframe
            Initial Dataframe
        
        Returns
        -------
        pandas.Dataframe
            Dataframe with the converted indices
        """
        temp_date = pd.to_datetime(data['time'])
        data = (
            data
            .set_index(temp_date)
            .drop(columns=['time', 'OPS'])
        )
        idx = pd.date_range(self.START + relativedelta.relativedelta(days=1), self.TODAY)
        data.index = pd.DatetimeIndex(data.index.date)
        data = (data.reindex(idx, fill_value=0))
        
        return data

    def getDataTest(self, market):
        """Get the test set making queries to InfluxDB 
        
        Parameters
        ----------
        market : str
            The market of interest
        
        Returns
        -------
        pandas.Dataframe, pandas.Dataframe
            The Dataframes of demand and supply
        """
        # Get the demand data from InfluxDB
        res = (
            self.client
            .query(
                f"SELECT * FROM demand{market} WHERE op='{self.op}' AND time >= '{self.START}'"
            )
            .raw
        )
        try:
            dem =(
                pd
                .DataFrame(
                    res['series'][0]['values'], 
                    columns = ['time', 'P', 'Q', 'OPS']
                )
            )
            dem = self.manageIndexTest(dem)
        except:
            dem = -1
        # Get the supply data from InfluxDB
        res = (
            self.client
            .query(
                f"SELECT * FROM supply{market} WHERE op='{self.op}' AND time >= '{self.START}'"
            )
            .raw
        )
        try:
            sup =(
                pd
                .DataFrame(
                    res['series'][0]['values'], 
                    columns = ['time', 'P', 'Q', 'OPS']
                )
            )
            sup = self.manageIndexTest(sup)
        except:
            sup = -1

        return dem, sup


    def runTest(self, data, label):
        """Method which runs the ARIMA model
        
        Parameters
        ----------
        data : pandas.DataFrame
            Input data for the ARIMA model
        label : str
            A label
        
        Returns
        -------
        numpy.ndarray
            Array of out of sample forecasts
        """
        X = data.values.astype('float32')
        
        model = ARIMA(
            X, 
            order=(0,1,0),
        )
        model_fit = model.fit(maxiter=200, disp=0, method='css')
        y = model_fit.forecast(H)[0][-1]
        
        if y < 0:
            y = 0
        return y 

    def predict(self):
        """Make the predictions.
        
        Returns
        -------
        List, List
            Prediction and validations for supply and demand data
        """
        pred = []
        for m in ['MGP', 'MI', 'MSD']:
            # Get the demand and supply curves
            d, s = self.getDataTest(m)
            # If the operator is not active in the market, getData 
            # returns -1, so the same -1 is appended to theprediction 
            # and it is managed later by the genetic algorithm
            if isinstance(s, int):
                pred.append(-1.0)
                pred.append(-1.0)
            else:
                pred.append(self.runTest(s.P, f'{m}pO'))
                pred.append(self.runTest(s.Q, f'{m}qO'))
            # If the operator is not active in the market, getData 
            # returns -1, so the same -1 is appended to the prediction 
            # and it is managed later by the genetic algorithm            
            
            if isinstance(d, int):
                pred.append(-1.0)
                pred.append(-1.0)
            else:
                pred.append(self.runTest(d.P, f'{m}pD'))
                pred.append(self.runTest(d.Q, f'{m}qD'))
        
        return np.asarray(pred)