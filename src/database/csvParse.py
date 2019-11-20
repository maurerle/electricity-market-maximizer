""" The parse has to convert the input CSV into a JSON file in order to load it on a
     MongoDB DataBase
"""
import sys
import pandas as pd
import logging
from pathlib import Path
from collections import defaultdict
import xlrd

if sys.version_info[:2] < (3, 7):
    raise RuntimeError("Python version >= 3.7 required.")

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s : %(message)s',
                    datefmt='%d/%m/%Y %H:%M ',
                    level=logging.DEBUG)

logger = logging.getLogger(__name__)

class ParseCsv():
    """A Parse class used to parse CSV file into json file for processing them on a MongoDB.
     Methods
    -------
    excel_to_dic

    """

    @staticmethod
    def excel_to_dic(path, header=None, skiprows=2, flag=False):

        """
        reads an excel file, located in a specific path, transforming it into a python dictionary

         :param path: path excel file path
         :param skiprows: list-like, int or callable, optional. Line numbers to skip (0-indexed) or number of lines
                    to skip (int) at the start of the file.

         :param header: int, list of int, Row number(s) to use as the column names, and the start of the data

         :param flag: Boolean, it is just a check if the file needs 3 rows to be skip. Default is False.

         :return a pandas.DataFrame

        """
        try:
            if flag:
                df = pd.read_excel(io=path, header=header, skiprows=3)
                df = df.drop([0])
                return df
            else:
                df = pd.read_excel(io=path, header=header, skiprows=skiprows)
                df = df.drop([0])
                return df
        except Exception as e:
            logging.error("Exception occurred", exc_info=True)



    @staticmethod
    def to_list_dict(df):
        """This method can process correctly all types of TERNA's file. """
        ag_dict_ = df.groupby([0])[[2, 1]].apply(lambda g: dict(map(tuple, g.values.tolist())))
        # Energy Balance: the total power used to meet the total load of italy, internal and foreign power.
        ls = []
        for _key, _value in ag_dict_.items():
            ag_dict = {}
            ag_dict['Data'] = _key.strftime('%Y%m%d')
            ag_dict['Ora'] = _key.strftime('%H')
            for key, value in _value.items():
                ag_dict[key] = value
            ls.append(ag_dict)
            
        return ls


    @staticmethod
    def join_dict(*args):
        """a method for joining dictionary with same keys: here, it takes all the possible dictionary generates by our
        model, and return a dictionary with key=date and values as many as values are present in all initial dictionary"""

        d = defaultdict(list)
        for d_ in args:
            for key, value in d_.items():
                d[key].append(value)
        return d