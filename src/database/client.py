from pymongo import MongoClient
import sys, os


# Connection to the MongoDB Server
mongoClient = MongoClient ('mongodb+srv://new-user:politomerda@cluster0-awyti.mongodb.net/test?retryWrites=true&w=majority')
# Connection to the database
db = mongoClient['InterProj']

# Menu definition
menu_actions = {
    '1': 'MGP',
    '2': 'MI',
    '3': 'MSD',
}
 
# Main menu
def main_menu():
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
    print("\n\t [0] Quit")
    choice = input(" >>  ")
    exec_menu(choice)
 
# Execute menu
def exec_menu(choice):
    os.system('cls' if os.name == 'nt' else 'clear')
    ch = choice.lower()
    if ch == '9': 
        back()
    if ch == '0':
        exit()
    try:
        date_hour_menu(menu_actions[ch])
    except KeyError:
        print("Invalid selection, please try again.\n")
        main_menu()
 
def date_hour_menu(market):
    print("  --------------- Market: "+ market +" ---------------\n")
    date = input("  Insert date (yyyymmdd) >>  ")
    hour = input("  Insert hour (hh) >>  ")

    collection = db[market]
    col = collection.find_one({"Data": date, 'Ora': hour})

    print("\n  ------------------ Output ------------------\n")
    for keys in col.keys(): 
        print ("  ", keys.ljust(50,'_'), col[keys])

    print("\n  Number of fields: ", len(col))

    print("\n\t [9] Back")
    print("\t [0] Quit")
    choice = input(" >>  ")
    exec_menu(choice)
    
# Back to main menu
def back():
    main_menu()
 
# Exit program
def exit():
    sys.exit()


if __name__ == "__main__":
    # Launch main menu
    main_menu()