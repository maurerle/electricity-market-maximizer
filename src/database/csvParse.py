from sys import version_info, dont_write_bytecode
from pandas import read_excel, ExcelFile
from numpy import max as nmax
from datetime import datetime
import pprint
from datetime import datetime
dont_write_bytecode = True

def ParseCsv(path):
    """Reads an excel file, located in a specific path, transforming it 
    into a python dictionary.
    
    Parameters
    ----------
    path : pathlib.Path
        path of the file to process
    header : optional
        by default None
    skiprows : int, optional
        number of rows to skip, by default 2
    
    Returns
    -------
    pandas.DataFrame
        dataframe containing the parsed data
    """
    load = 0
    xl = ExcelFile(path)
    sheets = xl.sheet_names
    for item in sheets:
        if 'Tot' in item and 'Terz' in item:
            d, l = process(path, item, 'TERZ', 2)
            load += l
            date = d
        if 'Sec' in item:
            d, l = process(path, item, 'SEC', 2)
            load += l
            date = d 

    return date, load


def process(path, sheet, label, skip):
    df = read_excel(io=path, sheet_name = sheet, skiprows=skip).dropna()
    try:
        df.columns = df.iloc[0]
        df = df.drop(0)
        for column in df.columns:
            if label in column:
                load = df[column].sum()
        date = str(df['DATA_ORA'][1]).split(' ')[0]
        date = datetime.strptime(date, '%Y-%m-%d')
        return date, load
    except KeyError:
        return process(path, sheet, label, skip+1)