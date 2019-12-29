from src.machinelearning.lstm import MGP
import getpass
import pandas as pd
import matplotlib.pyplot as plt

global user 
global passwd 

user = input('Insert SSH username:\n')
passwd = getpass.getpass(prompt='Insert SSH passwd:\n')

mgp = MGP(user, passwd)
#mgp.createSet(1543615200.0, 'IREN ENERGIA SPA')
data = mgp.createSet(0, 'IREN ENERGIA SPA')
#data = pd.read_csv('datasetTest.csv')
mgp.train(data)
#mgp.predict(data)