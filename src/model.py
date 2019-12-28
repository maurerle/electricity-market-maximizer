from src.machinelearning.lstm import MGP
import getpass

global user 
global passwd 

user = input('Insert SSH username:\n')
passwd = getpass.getpass(prompt='Insert SSH passwd:\n')

mgp = MGP(user, passwd)
#mgp.createSet()
mgp.manage()