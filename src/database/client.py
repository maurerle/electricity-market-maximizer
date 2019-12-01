from pymongo import MongoClient
import sys, os
import re


def databaseInit():
    """Initialize the connection to the database.

    Returns
    -------
    db : pymongo.database.Database
        the database to use
    """

    try:
        #client = MongoClient('mongodb+srv://new-user:politomerda@cluster0-awyti.mongodb.net/test?retryWrites=true&w=majority')
        client = MongoClient('localhost', 27017)
        db = client['InterProj']
        return db
    except Exception as e:
        print("Exception while connecting to the db: " + str(e))


def main_menu(db):
    """Main menu. The user can choose a market or quit the program
    
    Parameters
    ----------
    db : pymongo.database.Database
        the database to use
    """

    os.system('cls' if os.name == 'nt' else 'clear')
    
    header = "\
        ___  ___       ___ _ _            _   \n\
       /   \\/ __\\     / __\\ (_) ___ _ __ | |_ \n\
      / /\\ /__\\//    / /  | | |/ _ \\ '_ \\| __|\n\
     / /_// \\/  \\   / /___| | |  __/ | | | |_ \n\
    /___,'\\_____/   \\____/|_|_|\\___|_| |_|\\__|\n\n"

    print(header)
    print("   Welcome,\n")
    print("   Please choose the market you want:\n")
    print('\t [1] MGP (Mercato del giorno prima)')
    print('\t [2] MI (Mercato Infragiornaliero)')
    print('\t [3] MSD (Mercato per il Servizio di Dispacciamento)')
    print('\t [4] Terna')
    print("\n\t [0] Quit")
    choice = input(" >>  ")
    exec_menu(choice, db)

def exec_menu(choice, db):
    """Run the proper function according to the user's choice.
    
    Parameters
    ----------
    choice : str
        user's choice
    db : pymongo.database.Database
        the database to use
    """

    os.system('cls' if os.name == 'nt' else 'clear')
    
    menu_actions = {'1': 'MGP', '2': 'MI', '3': 'MSD', '4': 'Terna'}

    ch = choice.lower()

    if ch == '9': 
        # Go back to main menu
        main_menu(db)
    if ch == '0':
        # Quit the program
        sys.exit()

    try:
        date_hour_menu(db, menu_actions[ch])
    except KeyError:
        print("Invalid selection, please try again.\n")
        main_menu(db)
 
def date_hour_menu(db, market):
    """The user is asked to insert the date and hour for the desired output
    
    Parameters
    ----------
    db : pymongo.database.Database
        the database to use
    market : str
        market chosen by the user
    """

    print("  --------------- Market: "+ market +" ---------------\n")

    # Continue asking for a date until the format is not accepted
    while True:
        date = input("  Insert date (yyyymmdd) >>  ")
        if re.match(r'^[0-9]{8}$', date):
            break
        print('  Invalid date, please enter again!')

    # Continue asking for an hour until the format is not accepted
    while True:
        hour = input("  Insert hour (hh) >>  ")
        if re.match(r'^(0[1-9]|1[0-9]|2[0-4])$', hour):
            break
        print('  Invalid hour, please enter a value from 01 to 24!')

    getDocument(db, market, date, hour)

    print("\n\t [9] Back")
    print("\t [0] Quit")
    choice = input(" >>  ")
    exec_menu(choice, db)

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


if __name__ == "__main__":
    # Connect to database
    db = databaseInit()
    # Launch main menu
    main_menu(db)