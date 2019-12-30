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
#data = mgp.createSetTrain(0, 'IREN ENERGIA SPA')


data, future = mgp.createSetPred(1575158400, 'IREN ENERGIA SPA')

res = mgp.predict(data)
res = res[-1]
for i in future.index:
    future.loc[i, 'DLY_QTY'] = res
    
    data = data.append(future.loc[i])
    
    res = mgp.predict(data)
    res = res[-1]

    print(data)

#data = pd.read_csv('datasetTest.csv')
#mgp.train(data)
#mgp.predict(data)