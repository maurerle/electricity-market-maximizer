import pandas as pd 
from influxdb import InfluxDBClient
from datetime import datetime, date
from dateutil import relativedelta
import numpy as np
from .genetic import Genetic

client = InfluxDBClient(
    '172.28.5.1', 
    8086, 
    'root', 
    'root', 
    'PublicBids'
)

def optimize(target, limit):
    """Run the optimization process of the genetic algorithm.
    
    Parameters
    ----------
    target : str
        Operator whose profit has to be optimized
    limit : float
        [description]
    
    Returns
    -------
    JSON str, numpy.ndarray
		Optimized profits, best solution
    """
    dummy_now = (
        date.today()- relativedelta.relativedelta(months=1) + relativedelta.relativedelta(days=1)
    ).strftime('%d/%m/%Y')
    TODAY = datetime.strptime(dummy_now, '%d/%m/%Y')

    client = InfluxDBClient('172.28.5.1', 8086, 'root', 'root', 'PublicBids')
 
    res = client.query(f"SELECT * FROM predictions").raw

    data = pd.DataFrame(res['series'][0]['values'], columns = res['series'][0]['columns']).set_index('op')
    genetic = Genetic(target, data, TODAY, limit)

    return genetic.run()