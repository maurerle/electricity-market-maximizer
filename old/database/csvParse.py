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

class ParseCsv:
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
    def to_list_dict(df, flag):
        """This method can process correctly all types of TERNA's file. """

        logging.info(f'A dictionary for file {flag} will be created.')
        if flag == 'TL' or flag == 'M':
            # it is not clear what does Market Load represent (?) ask to Huang
            tl_dict = {}
            for i in range(len(df[0])):
                if i * 4 < (len(df[0])):
                    if flag == 'TL':
                        tl_dict[df[0][i * 4 + 1].strftime("%d/%m/%Y - %H:%M")] = {'Total Load': df[1][i * 4 + 1]} # OK but until 23:0
                    else:
                        tl_dict[df[0][i * 4 + 1].strftime("%d/%m/%Y - %H:%M")] = {'Market Load': df[1][i * 4 + 1]}

            return tl_dict

        if flag == 'AG' or flag == 'E':  # ok
            ag_dict_ = df.groupby([0])[[2, 1]].apply(lambda g: dict(map(tuple, g.values.tolist()))).to_dict()
            ag_dict = {}
            if flag == 'AG':
                #  Actual Generation: the total power generated by Italy
                for key, value in ag_dict_.items():
                    ag_dict[key.strftime('%d/%m/%Y - %H:%M')] = {'Actual Generation': value}
                return ag_dict
            else:
                # Energy Balance: the total power used to meet the total load of italy, internal and foreign power.
                ls = []
                for key, value in ag_dict_.items():
                    ls.append({'Data': key.strftime('%d/%m/%Y'), 'Ora': key.strftime('%H:%M'), 'Energy Balance': value})
                    #ag_dict_E = {'Data': key.strftime('%d/%m/%Y'), 'Ora': key.strftime('%H:%M'), 'Energy Balance': value}
                return ls
            # ag_dict[key.strftime('%d/%m/%Y - %H:%M')] = {'Energy Balance': value}

        elif flag == 'FS' or flag == 'FP':  #
            f_dict_1 = df.groupby([0])[[1, 2]].apply(lambda g: dict(map(tuple, g.values.tolist()))).to_dict()
            f_dict_1_ = {}
            for key, value in f_dict_1.items():
                f_dict_1_[key] = {'Import': value}

            f_dict_2 = df.groupby([0])[[1, 3]].apply(lambda g: dict(map(tuple, g.values.tolist()))).to_dict()
            f_dict_2_ = {}
            for key, value in f_dict_2.items():
                f_dict_2_[key] = {'Export': value}

            f_dict_3 = df.groupby([0])[[1, 4]].apply(lambda g: dict(map(tuple, g.values.tolist()))).to_dict()
            f_dict_3_ = {}

            f_dd_ = defaultdict(list)
            if flag == 'FS':
                # Scheduled Foreign Exchange
                for key, value in f_dict_3.items():
                    f_dict_3_[key] = {'SFE (MW)': value}
                for d in (f_dict_1_, f_dict_2_, f_dict_3_):
                    for key, value in d.items():
                        f_dd_[key].append(value)

                f_dict = {}
                for key, value in f_dd_.items():
                    f_dict[key.strftime("%d/%m/%Y, %H:%M")] = {'Scheduled Foreign Exchange': value}
                return f_dict
            else:
                #  Foreign Physical Flow
                for key, value in f_dict_3.items():
                    f_dict_3_[key] = {'FPF (MW)': value}
                for d in (f_dict_1_, f_dict_2_, f_dict_3_):
                    for key, value in d.items():
                        f_dd_[key].append(value)

                f_dict = {}
                for key, value in f_dd_.items():
                    f_dict[key.strftime("%d/%m/%Y, %H:%M")] = {'Foreign Physical Flow': value}
                return f_dict

            f_dd_ = defaultdict(list)
            for d in (f_dict_1_, f_dict_2_, f_dict_3_):
                for key, value in d.items():
                    f_dd_[key].append(value)

            f_dict = {}
            for key, value in f_dd_.items():
                f_dict[key.strftime("%d/%m/%Y, %H:%M")] = {'LOLLO': value}
            return f_dict

        elif flag == 'IP' or flag == 'IS':
            i_dict_1 = df.groupby([0])[[0, 1]].apply(lambda g: dict(map(tuple, g.values.tolist()))).to_dict()
            i_dict_1_ = {}
            for key, value in i_dict_1.items():
                i_dict_1_[key] = {'Zone from': zone for zone in value.values()}

            i_dict_2 = df.groupby([0])[[0, 2]].apply(lambda g: dict(map(tuple, g.values.tolist()))).to_dict()
            i_dict_2_ = {}
            for key, value in i_dict_2.items():
                i_dict_2_[key] = {'Zone to': zone for zone in value.values()}

            f_dd = defaultdict(list)
            if flag == 'IP':
                i_dict_3 = df.groupby([0])[[0, 3]].apply(lambda g: dict(map(tuple, g.values.tolist()))).to_dict()
                i_dict_3_ = {}
                for key, value in i_dict_3.items():
                    # Physical Internal Flow
                    i_dict_3_[key] = {'PIF (MW)': zone for zone in value.values()}
                for d in (i_dict_1_, i_dict_2_, i_dict_3_):
                    for key, value in d.items():
                        f_dd[key].append(value)
                i_dict = {}
                for key, value in f_dd.items():
                    i_dict[key.strftime("%d/%m/%Y, %H:%M")] = {'Internal Physical Flow': value}
                return i_dict
            else:
                i_dict_3 = df.groupby([0])[[0, 3]].apply(lambda g: dict(map(tuple, g.values.tolist()))).to_dict()
                i_dict_3_ = {}
                for key, value in i_dict_3.items():
                    # Scheduled Internal Exchange
                    i_dict_3_[key] = {'SIE (MW)': zone for zone in value.values()}
                for d in (i_dict_1_, i_dict_2_, i_dict_3_):
                    for key, value in d.items():
                        f_dd[key].append(value)
                i_dict = {}
                for key, value in f_dd.items():
                    i_dict[key.strftime("%d/%m/%Y, %H:%M")] = {'Scheduled Internal Exchange': value}
                return i_dict

    """TO DO: if needed, all other if/elif branch must be update as the branch of Energy Balance, because we want a list
    of dictionary, such that each elements of the list is a document just with key ['DATA', 'TIME', 'VALUE']"""

    @staticmethod
    def find_name(path):
        """This method allows to map each file name in the directory to a flag, in order to know which parser must be
         called.

         :parameter : path of the directory where the files are

         :return flist: a list with all the file excel's paths.
         dic_name: a dictionary with key a flag (specific for each types of
         file) and as value the index of the flist associated with that flag"""

        p = Path(path)
        #flist = [x for x in p.glob('*.xlsx')]
        dict_name = {}
        #for i in range(len(flist)):

        fname = p.stem
        flag = ""
        if fname[0] == 'T':
            flag = 'TL'

        elif fname[0] == 'M':
            flag = 'M'

        elif fname[0] == 'E':
            flag = 'E'

        elif fname[0] == 'A':
            flag = 'AG'

        elif fname[0:9] == 'Foreign_S':
            flag = 'FS'

        elif fname[0:9] == 'Foreign_P':
            flag = 'FP'

        elif fname[0:10] == 'Internal_P':
            flag = 'IP'

        elif fname[0:10] == 'Internal_S':
            flag = 'IS'

        return flag

    @staticmethod
    def join_dict(*args):
        """a method for joining dictionary with same keys: here, it takes all the possible dictionary generates by our
        model, and return a dictionary with key=date and values as many as values are present in all initial dictionary"""

        d = defaultdict(list)
        for d_ in args:
            for key, value in d_.items():
                d[key].append(value)
        return d




if __name__ == "__main__":

    path = '/Users/gianpiodomiziani/Desktop/data/Energy_balance19112019.xlsx'


    df_E = ParseCsv.excel_to_dic(path)
    dict_E = ParseCsv.to_dict(df_E, 'E')