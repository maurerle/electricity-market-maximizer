import pandas as pd

data = pd.read_csv('Generator.csv')
qty = 0
for i in range(data.shape[0]):
    if 'IREN' in data.iloc[i]['operator']:
        qty+=data.iloc[i]['max_quantity']
print(qty)