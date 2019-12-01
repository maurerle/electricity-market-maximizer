from sys import version_info, dont_write_bytecode
from pandas import read_excel
from numpy import max as nmax
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
    if version_info[:2] < (3, 7):
        raise RuntimeError("Python version >= 3.7 required.")

    @classmethod
    def excel_to_dic(cls, path, header=None, skiprows=3):
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
        
        df = read_excel(io=path, header=header, skiprows=skiprows)
        cls.date = path[-13:-5]
        cls.date = f'{cls.date[-4:]}{cls.date[2:4]}{cls.date[0:2]}'

        return df

    @classmethod
    def to_list_dict(cls, df, field):
        """Create a list of dictionaries. Each dictionary will be uploaded
        on the database as a single document.
        
        Parameters
        ----------
        df : pandas.DataFrame
            dataframe created by the excel_to_dict() method
        field: str
            download field of the Terna website
        
        Returns
        -------
        list
            dict list to be updated on the database
        """
        ls = []
        if field == 'EnBal':
            ag_dict_ = df.groupby([0])[[2, 1]].apply(
                lambda g: dict(map(tuple, g.values.tolist()))
            )
            for _key, _value in ag_dict_.items():
                ag_dict = {}
                ag_dict['Data'] = _key.strftime('%Y%m%d')
                ag_dict['Ora'] = _key.strftime('%H')
                for key, value in _value.items():
                    ag_dict[key] = value
                ls.append(ag_dict)
        elif field == 'ToLo':
            for i in range(0,len(df),4):
                ls.append(
                    {
                        'TotalLoad':nmax(df.iloc[i:i+4,1]),
                        'TotalLoadForecast':nmax(df.iloc[i:i+4,2]),
                        'Ora':df.iloc[i,0].strftime('%H'),
                        'Data':cls.date
                    }
                )
        elif field == 'MaLo':
            for i in range(0,len(df),4):
                ls.append(
                    {
                        'MarketLoad':nmax(df.iloc[i:i+4,1]),
                        'MarketLoadForecast':nmax(df.iloc[i:i+4,2]),
                        'Ora':df.iloc[i,0].strftime('%H'),
                        'Data':cls.date
                    }
                )
        else:
            ag_dict_ = df.groupby([1]).apply(
                lambda g: dict(map(tuple, g.values.tolist()))
            )
            for _key, _value in ag_dict_.items():
                if _key < datetime(2017, 2, 1):
                    return None

                ag_dict = {}
                ag_dict['Data'] = _key.strftime('%Y%m%d')
                ag_dict['Ora'] = _key.strftime('%H')
                for key, value in _value.items():
                    ag_dict[f'ResSec_{key}'] = value

                ls.append(ag_dict)
                if ag_dict['Ora'] == '23':
                    break
        
        for item in ls:
            item['Timestamp'] = datetime.strptime(
                    f"{item['Data']}:{item['Ora']}", 
                    '%Y%m%d:%H'
                ).timestamp()
            
        return ls 