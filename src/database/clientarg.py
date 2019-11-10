import argparse
import re
import sys
from pymongo import MongoClient


def parse_args(parser):
    """Parse the arguments.
    
    Parameters
    ----------
    parser : argparse.ArgumentParser
        argument parser

    Returns
    -------
    argparse.Namespace
        parsed arguments
    """

    parser.add_argument("-m", "--market", action="store", dest="market", choices=['MGP','MI','MSD'],
                        help="The market you are interested in")
    parser.add_argument("-d", "--date", action="store", dest="date", type=valid_date,
                        help="Date in YYYYMMDD format")
    parser.add_argument("-hr", "--hour", action="store", dest="hour", type=valid_hour,
                        help="Hour in HH format (from 01 to 24)")
    parser.add_argument("-rm", "--remove", action="store_true", dest="remove", default=False,
                        help="To remove database or collection. If market is not passed, remove all the collections")

    return parser.parse_args()

def databaseInit():
    """Initialize the connection to the database.

    Returns
    -------
    db : pymongo.database.Database
        the database to use
    """

    try:
        client = MongoClient('mongodb+srv://new-user:politomerda@cluster0-awyti.mongodb.net/test?retryWrites=true&w=majority')
        db = client['InterProj']
    except Exception as e:
        print("Exception while connecting to the db: " + str(e))
    
    return db

def getDocument(db, market, date, hour):
    """Retrive the document from the database and print it.
    
    Parameters
    ----------
    db : the database to use
        the database to use
    market : str
        market chosen by the user
    date : str
        date chosen by the user
    hour : str
        hour chosen by the user
    """

    collection = db[market]
    doc = collection.find_one({"Data": date, 'Ora': hour})

    if doc == None:
        print("\n  No data found for " + date + " at " + hour)
    else:
        print("\n  ------------------ Output ------------------\n")
        for keys in doc.keys(): 
            print ("  ", keys.ljust(50,'_'), doc[keys])

        print("\n  Number of fields: ", len(doc))

def valid_date(s):
    """Check if the input date is valid.
    
    Parameters
    ----------
    s : str
        string date

    Returns
    -------
    str
        the entire matched string.

    Raises
    ------
    argparse.ArgumentTypeError
        If the date format is not valid.
    """

    try:
        return re.match(r'^[0-9]{8}$', s).group(0)
    except:
        msg = "Not a valid date: '{0}'. Required format is YYYYMMDD".format(s)
        raise argparse.ArgumentTypeError(msg)

def valid_hour(s):
    """Check if the input hour is valid.
    
    Parameters
    ----------
    s : str
        string hour

    Returns
    -------
    str
        the entire matched string.

    Raises
    ------
    argparse.ArgumentTypeError
        If the hour format is not valid.
    """

    try:
        return re.match(r'^(0[1-9]|1[0-9]|2[0-4])$', s).group(0)
    except:
        msg = "Not a valid hour: '{0}'. Required format is HH (from 01 to 24)".format(s)
        raise argparse.ArgumentTypeError(msg)


# Initialize the database
db = databaseInit()

# Some usage example for the help message
example_text = '''example:

    python clientarg.py -m MGP -d 20191107 -hr 11
    python clientarg.py -m MSD -rm
    python clientarg.py -rm'''
parser = argparse.ArgumentParser(description='Database client', 
                                 epilog=example_text,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)

args = parse_args(parser)

market = args.market
date = args.date
hour = args.hour
remove = args.remove

# if no argument passed
if not len(sys.argv) > 1:
    parser.print_help()
    sys.exit() 

# Remove mode
if remove:
    # Date and/or hour must not be passed when removing
    if date or hour:
        raise parser.error("Not a valid --remove usage. Must be --remove [-m {MGP,MI,MSD}]")
    
    # If a market is passed, it will be the only one to be removed
    if market:
        choice = input("Are you sure you want to remove '{0}'? (y/n) >> ".format(market))
        if choice == 'y':
            db.drop_collection(market)
            sys.exit()
        else:
            sys.exit()
    # If no market is passed, every market will be removed
    else:
        choice = input("Are you sure you want to remove every collection? (y/n) >> ")
        if choice == 'y':
            for collection in db.collection_names():
                db.drop_collection(collection)
            sys.exit()
        else:
            sys.exit()

# Visualization mode
if not remove:
    if market and date and hour:
        getDocument(db, market, date, hour)
    else:
        raise parser.error("market, date and hour must be passed for an output")
