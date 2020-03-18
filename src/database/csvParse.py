from sys import version_info, dont_write_bytecode
from pandas import read_excel
from numpy import max as nmax
from datetime import datetime
import pprint
from datetime import datetime
dont_write_bytecode = True

class ParseCsv():
    """.csv parser to process the .xlsx file and store them in a MongoDB 
    database.
    
    Methods
    -------
    excel_to_dic(path, header=None, skiprows=2, flag=False)
    to_list_dict(df, field)

    Raises
    ------
    RuntimeError
        check if the python version is the 3.7 one
    """

    @classmethod
    def excel_to_dic(cls, path, header=None, skiprows=4):
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
        df = read_excel(io=path, header=header, skiprows=skiprows).dropna()
        df.columns = df.iloc[0]
        df = df.drop(0)
        load = df['FABBISOGNO'].sum()
        date = str(df['DATA_ORA'][1]).split(' ')[0]
        date = datetime.strptime(date, '%Y-%m-%d')

        return date, load