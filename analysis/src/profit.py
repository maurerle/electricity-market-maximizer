import numpy as np
from .intersection import intersection
from influxdb import InfluxDBClient


def secRes(today):
    """Query for the reserve thresholds.
    
    Returns
    -------
    list
        Secondary reserve thresholds
    """
    client = InfluxDBClient(
        'localhost', 
        8086, 
        'root', 
        'root', 
        'PublicBids'
    )
    res = (
        client
        .query(f"SELECT * FROM STRes WHERE time = '{today}'")
        .raw
    )
    return res['series'][0]['values'][0][1]


def computeClearing1(off, bid):
    """Simplified computation of the clearing price for MGP/MI
    
    Parameters
    ----------
    off : pandas.DataFrame
        Offers
    bid : pandas.DataFrame
        Bids
    
    Returns
    -------
    float
        Clearing price
    """
    sup = off[(off >= 0).any(1)]
    dem = bid[(bid >= 0).any(1)]
    # Sort the prices
    sup = off.sort_values('P', ascending=True)
    dem = bid.sort_values('P', ascending=False)
    # Cumulative sums of quantity
    sup_cum = np.cumsum(sup['Q'])
    dem_cum = np.cumsum(dem['Q'])
    # Find the curves intersection
    clearing = intersection(
        sup_cum.values, 
        sup.P.values, 
        dem_cum.values, 
        dem.P.values
    )[1][0]

    return clearing

def computeClearing2(off, bid, today):
    """Simplified computation of the 'dummy' clearing price for MSD
    
    Parameters
    ----------
    off : pandas.DataFrame
        Offers
    bid : pandas.DataFrame
        Bids
    
    Returns
    -------
    float, float
        Clearing price of demand and supply
    """
    sup = off[(off >= 0).any(1)]
    dem = bid[(bid >= 0).any(1)]
    # Sort the prices
    sup = off.sort_values('P', ascending=True)
    dem = bid.sort_values('P', ascending=False)
    # Cumulative sums of quantity
    sup_cum = np.cumsum(sup['Q'])
    dem_cum = np.cumsum(dem['Q'])
    # Get the MSD quantity threshold
    th = secRes(today)
    # Create the th curve
    x_th = np.array([th, th])
    y_th = np.array([0, np.max(sup.P.values)])

    clearingD = intersection( 
        dem_cum.values, 
        dem.P.values, 
        x_th,
        y_th
    )[1][0]
    clearingS = intersection( 
        sup_cum.values, 
        sup.P.values,
        x_th,
        y_th
    )[1][0]
    
    return clearingD, clearingS

def getProfit1(sup, dem, pun, target):
    """Evaluate the profit in MGP/MI market.
    
    Parameters
    ----------
    sup : pandas.DataFrame
        Supply curve
    dem : pandas.DataFrame
        Demand curve
    pun : float
        National Single Price
    
    Returns
    -------
    float
        Profit
    """
    if sup.loc[target].P > pun:
        # Rejected bid for the supply
        Qsup = 0.0
    else:
        # Accepted bid for the supply
        Qsup = sup.loc[target].Q
    if dem.loc[target].P < pun:
        # Rejected bid for the demand
        Qdem = 0.0
    else:
        # Accepted bid for the demand
        Qdem = dem.loc[target].Q
    
    return (Qsup - Qdem)*pun

def getProfit2(sup, dem, punS, punD, target):
    """Evaluate the profit in MSD market.
    
    Parameters
    ----------
    sup : pandas.DataFrame
        Supply curve
    dem : pandas.DataFrame
        Demand curve
    punS : float
        National Single Price of supply 
    punD : float
        National Single Price of demand 
    
    Returns
    -------
    float
        Profit
    """
    if sup.loc[target].P > punS:
        # Rejected bid for the supply
        Qsup = 0.0
    else:
        # Accepted bid for the supply
        Qsup = sup.loc[target].Q
    Psup = sup.loc[target].P
    
    if dem.loc[target].P < punD:
        # Rejected bid for the demand
        Qdem = 0.0
    else:
        # Accepted bid for the demand
        Qdem = dem.loc[target].Q
    Pdem = dem.loc[target].P
    return Qsup*Psup - Qdem*Pdem


def getNewProfit(data, target, today, individual):
    dem1 = (
        data[['MGPqD','MGPpD']]
        .rename(columns={'MGPqD':'Q', 'MGPpD':'P'})
    )
    sup1 = (
        data[['MGPqO','MGPpO']]
        .rename(columns={'MGPqO':'Q', 'MGPpO':'P'})
    )
    dem2 = (
        data[['MIqD','MIpD']]
        .rename(columns={'MIqD':'Q', 'MIpD':'P'})
    )
    sup2 = (
        data[['MIqO','MIpO']]
        .rename(columns={'MIqO':'Q', 'MIpO':'P'})
    )
    dem3 = (
        data[['MSDqD','MSDpD']]
        .rename(columns={'MSDqD':'Q', 'MSDpD':'P'})
    )
    sup3 = (
        data[['MSDqO','MSDpO']]
        .rename(columns={'MSDqO':'Q', 'MSDpO':'P'})
    )

    sup1.loc[target] = [individual[0], individual[1]]
    dem1.loc[target] = [individual[2], individual[3]]
    sup2.loc[target] = [individual[4], individual[5]]
    dem2.loc[target] = [individual[6], individual[7]]
    sup3.loc[target] = [individual[8], individual[9]]
    dem3.loc[target] = [individual[10], individual[11]]

    # Set the 0 demanded price as the default one
    dem1.P = dem1.P.replace(0, 3000)
    dem2.P = dem2.P.replace(0, 3000)
    dem3.P = dem3.P.replace(0, 3000)
    # Determine the clearing price for MGP and MI
    try:
        pun1 = computeClearing1(sup1, dem1)
        profit_mgp = getProfit1(sup1, dem1, pun1, target)
    except:
        print('Error')
    try:
        pun2 = computeClearing1(sup2, dem2)
        profit_mi = getProfit1(sup2, dem2, pun2, target)
    except:
        print('Error')
    try:
        pun3d, pun3s = computeClearing2(sup3, dem3, today)
        profit_msd = getProfit2(sup3, dem3, pun3s, pun3d, target)
    except:
        print('Error')
    profit = profit_mgp + profit_mi + profit_msd

    return profit_mgp, profit_mi, profit_msd, profit




def getProfit(data, target, today):
    dem1 = (
        data[['MGPqD','MGPpD']]
        .rename(columns={'MGPqD':'Q', 'MGPpD':'P'})
    )
    sup1 = (
        data[['MGPqO','MGPpO']]
        .rename(columns={'MGPqO':'Q', 'MGPpO':'P'})
    )
    dem2 = (
        data[['MIqD','MIpD']]
        .rename(columns={'MIqD':'Q', 'MIpD':'P'})
    )
    sup2 = (
        data[['MIqO','MIpO']]
        .rename(columns={'MIqO':'Q', 'MIpO':'P'})
    )
    dem3 = (
        data[['MSDqD','MSDpD']]
        .rename(columns={'MSDqD':'Q', 'MSDpD':'P'})
    )
    sup3 = (
        data[['MSDqO','MSDpO']]
        .rename(columns={'MSDqO':'Q', 'MSDpO':'P'})
    )


    # Set the 0 demanded price as the default one
    dem1.P = dem1.P.replace(0, 3000)
    dem2.P = dem2.P.replace(0, 3000)
    dem3.P = dem3.P.replace(0, 3000)
    # Determine the clearing price for MGP and MI
    try:
        pun1 = computeClearing1(sup1, dem1)
        profit_mgp = getProfit1(sup1, dem1, pun1, target)
    except:
        print('Error')
    try:
        pun2 = computeClearing1(sup2, dem2)
        profit_mi = getProfit1(sup2, dem2, pun2, target)
    except:
        print('Error')
    try:
        pun3d, pun3s = computeClearing2(sup3, dem3, today)
        profit_msd = getProfit2(sup3, dem3, pun3s, pun3d, target)
    except:
        print('Error')
    profit = profit_mgp + profit_mi + profit_msd

    return profit_mgp, profit_mi, profit_msd, profit