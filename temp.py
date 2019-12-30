import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
bid = pd.read_csv('bid.csv', index_col='Unnamed: 0')
mgp = pd.read_csv('mgp.csv', index_col='Timestamp')
bid.index = pd.to_datetime(bid.index)
mgp.index = pd.to_datetime(mgp.index)
#pd.set_option('display.max_rows', None)

def getDifferences():
    df = pd.merge(bid, mgp, left_index=True, right_index=True, how='outer', indicator='Exist')
    df = df.loc[df['Exist'] == 'right_only']
    
    for i in df.index:
        if i.strftime('%m-%Y') == datetime.now().strftime('%m-%Y'):
            return i
x = [1, 2, 3, 4]
print(x[-1])           

temp = mgp[mgp.index>=getDifferences()]

bid = bid.drop(columns=['DLY_AWD_QTY', 'TYPE', 'DLY_PRICE', 'DLY_AWD_PRICE'])

merged = pd.merge(mgp, bid, left_index=True, right_index=True)
cnt = 5
for i in temp.index:
    # cnt = predictions
    temp.loc[i, 'DLY_QTY'] = cnt
    
    print(merged)
    merged = merged.append(temp.loc[i])
    
    # scale and predict
    print(merged)

    cnt+=1