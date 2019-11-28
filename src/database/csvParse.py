from sys import version_info, dont_write_bytecode
from pandas import read_excel
import logging
from pathlib import Path
from collections import defaultdict
import xlrd

dont_write_bytecode = True

class ParseCsv():
    """.csv parser to process the .xlsx file and store them in a MongoDB 
    database.
    
    Methods
    -------
    excel_to_dic(path, header=None, skiprows=2, flag=False)
    to_list_dict(df)

    Raises
    ------
    RuntimeError
        check if the python version is the 3.7 one
    """
    def __init__(self):
        if version_info[:2] < (3, 7):
    
            raise RuntimeError("Python version >= 3.7 required.")


    @staticmethod
    def excel_to_dic(path, header=None, skiprows=2, flag=False):
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
        flag : bool, optional
            by default False
        
        Returns
        -------
        pandas.DataFrame
            dataframe containing the parsed data
        """
        if flag:
            df = read_excel(io=path, header=header, skiprows=3)
            df = df.drop([0])
            
            return df
        else:
            df = read_excel(io=path, header=header, skiprows=skiprows)
            df = df.drop([0])
            
            return df


    @staticmethod
    def to_list_dict(df, flag):
        """Create a list of dictionaries. Each dictionary will be uploaded
        on the database as a single document.
        
        Parameters
        ----------
        df : pandas.DataFrame
            dataframe created by the excel_to_dict() method.
        
        Returns
        -------
        list
            dict list to be updated on the database
        """
        if flag == 'E':
            ag_dict_ = df.groupby([0])[[2, 1]].apply(
                lambda g: dict(map(tuple, g.values.tolist()))
            )
        elif flag == 'SR':
            ag_dict_ = df.groupby([1]).apply(
                lambda g: dict(map(tuple, g.values.tolist()))
            )
            
        ls = []
        for _key, _value in ag_dict_.items():
            ag_dict = {}
            ag_dict['Data'] = _key.strftime('%Y%m%d')
            ag_dict['Ora'] = _key.strftime('%H')
            for key, value in _value.items():
                ag_dict[key] = value
            ls.append(ag_dict)
            
        return ls