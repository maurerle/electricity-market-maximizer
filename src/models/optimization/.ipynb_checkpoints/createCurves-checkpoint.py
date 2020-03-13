import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mergeDaily(df):
    merged = pd.DataFrame(columns=['Q', 'P', 'Q_AWD', 'P_AWD', 'OP'])
    cnt = 0
    for op in df['OPERATORE'].unique():
        temp = df.where(df['OPERATORE']==op).dropna()

        new = pd.DataFrame(columns=['Q', 'P', 'Q_AWD', 'P_AWD', 'OP'])
        q = np.sum(temp['QUANTITY_NO'])
        q_awd = np.sum(temp['AWARDED_QUANTITY_NO'])
        p = np.mean(temp['ENERGY_PRICE_NO'])
        p_awd = np.mean(temp['AWARDED_PRICE_NO'])
        new.loc[cnt] = [q, p, q_awd, p_awd, op]
        cnt+=1
        merged = pd.concat([merged, new], axis= 0)
    
    return merged


# +
def createOffCurve(df):
    prices = df['ENERGY_PRICE_NO'].unique()
    curve = pd.DataFrame(columns=['Q', 'P'])
    cnt = 0
    for p in prices:
        new = pd.DataFrame(columns=['Q', 'P'])
        temp = df.where(df['ENERGY_PRICE_NO']==p).dropna()
        q = np.sum(temp['QUANTITY_NO'])
        new.loc[cnt] = [q, p]
        cnt+=1
        curve = pd.concat([curve, new], axis= 0)
    
    curve = curve.sort_values('P')
    
    return(curve)

def createDemCurve(df):
    prices = df['ENERGY_PRICE_NO'].unique()
    curve = pd.DataFrame(columns=['Q', 'P'])
    cnt = 0
    for p in prices:
        new = pd.DataFrame(columns=['Q', 'P'])
        temp = df.where(df['ENERGY_PRICE_NO']==p).dropna()
        q = np.sum(temp['QUANTITY_NO'])
        
        if p == 0:
            new.loc[cnt] = [q, 3000]
        else:
            new.loc[cnt] = [q, p]
        
        cnt+=1
        curve = pd.concat([curve, new], axis= 0)
    
    curve = curve.sort_values('P', ascending=False)
    
    return(curve)
# -








# +
data = (
    pd
    .read_csv('data.csv')
    .drop(columns=[
        'MARKET_CD', 
        'UNIT_REFERENCE_NO', 
        'MARKET_PARTECIPANT_XREF_NO',
        'BID_OFFER_DATE_DT',
        'TRANSACTION_REFERENCE_NO',
        'MERIT_ORDER_NO',
        'PARTIAL_QTY_ACCEPTED_IN',
        'ADJ_QUANTITY_NO',
        'ADJ_ENERGY_PRICE_NO',
        'GRID_SUPPLY_POINT_NO',
        'SUBMITTED_DT',
        'BALANCED_REFERENCE_NO',
    ])
)

#
acc = (
    data
    #.where(data['ZONE_CD'].isin(['CNOR','CSUD','NORD','SARD','SICI','SUD','AUST','CORS','COAC','BRNN','FOGN','FRAN', 'GREC', 'MFTV', 'PRGP','ROSN','SLOV','SVIZ','MALT','XFRA']))
    #.where(data['OPERATORE']!='Bilateralista')
    .where(data['INTERVAL_NO']==5)
    .where(data['STATUS_CD'].isin(['ACC', 'REJ']))
    #.where(data['BILATERAL_IN']!=1)
    .dropna()
)
off = acc.where(acc['PURPOSE_CD']=='OFF').dropna().drop(columns='PURPOSE_CD')
bid = acc.where(acc['PURPOSE_CD']=='BID').dropna().drop(columns='PURPOSE_CD')

# +
#off = pd.DataFrame({'Q':off['QUANTITY_NO'],'P':off['ENERGY_PRICE_NO']})
#bid = pd.DataFrame({'Q':bid['QUANTITY_NO'],'P':bid['ENERGY_PRICE_NO']})
# -

bid['ENERGY_PRICE_NO'] = bid['ENERGY_PRICE_NO'].replace(0, 3000)

off = createOffCurve(off)
bid = createDemCurve(bid)

off_c = np.cumsum(off['Q'])
bid_c = np.cumsum(bid['Q'])

# +
plt.figure(figsize=(9,8))
plt.plot(bid_c, bid.P, label='Demand', linewidth=3)
plt.plot(off_c, off.P, label='Supply', linewidth=3)

plt.ylim(0,70)
#plt.xlim(15000)
#plt.yticks([46.18], labels=['$P_{cl}$'])
plt.yticks(np.arange(0, 600, 50))
#plt.xticks([24978.921, 25918.921], labels=['$Q_{Off}$', '$Q_{Bid}$'])
plt.axhline(y=46.18, linestyle = '--', color='k', linewidth=1.2)
plt.xticks(np.arange(0, 121000, 12000))
plt.vlines(x=24978.921, ymin=0, ymax=46.18, linestyle = '--', color='k', linewidth=1.2)
plt.vlines(x=25918.921, ymin=0, ymax=46.18, linestyle = '--', color='k', linewidth=1.2)
plt.grid()
plt.legend()
# -





















# +
data = (
    pd
    .read_csv('data.csv')
    .drop(columns=[
        'MARKET_CD', 
        'UNIT_REFERENCE_NO', 
        'MARKET_PARTECIPANT_XREF_NO',
        'BID_OFFER_DATE_DT',
        'TRANSACTION_REFERENCE_NO',
        'MERIT_ORDER_NO',
        'PARTIAL_QTY_ACCEPTED_IN',
        'ADJ_QUANTITY_NO',
        'ADJ_ENERGY_PRICE_NO',
        'GRID_SUPPLY_POINT_NO',
        'SUBMITTED_DT',
        'BALANCED_REFERENCE_NO',
    ])
)

#
acc = (
    data
    .where(data['ZONE_CD'].isin(['CNOR','CSUD','NORD','SARD','SICI','SUD','AUST','CORS','COAC','BRNN','FOGN','FRAN', 'GREC', 'MFTV', 'PRGP','ROSN','SLOV','SVIZ','MALT','XFRA']))
    #.where(data['OPERATORE']!='Bilateralista')
    .where(data['INTERVAL_NO']==5)
    .where(data['STATUS_CD'].isin(['ACC', 'REJ']))
    #.where(data['BILATERAL_IN']!=1)
    .dropna()
)
off = acc.where(acc['PURPOSE_CD']=='OFF').dropna().drop(columns='PURPOSE_CD')
bid = acc.where(acc['PURPOSE_CD']=='BID').dropna().drop(columns='PURPOSE_CD')
# -

off = pd.DataFrame({'Q':off['QUANTITY_NO'],'P':off['ENERGY_PRICE_NO']})
bid = pd.DataFrame({'Q':bid['QUANTITY_NO'],'P':bid['ENERGY_PRICE_NO']})

bid['P'] = bid['P'].replace(0, 3000)

off = off.sort_values('P')
bid = bid.sort_values('P', ascending=False)

off_c = np.cumsum(off['Q'])
bid_c = np.cumsum(bid['Q'])

# +
plt.figure(figsize=(9,8))
plt.plot(bid_c, bid.P, label='Demand', linewidth=3)
plt.plot(off_c, off.P, label='Supply', linewidth=3)

plt.ylim(0,200)
#plt.xlim(15000)
#plt.yticks([46.18], labels=['$P_{cl}$'])
plt.yticks(np.arange(0, 600, 50))
#plt.xticks([24978.921, 25918.921], labels=['$Q_{Off}$', '$Q_{Bid}$'])
plt.axhline(y=46.18, linestyle = '--', color='k', linewidth=1.2)
plt.xticks(np.arange(0, 121000, 12000))
plt.vlines(x=24978.921, ymin=0, ymax=46.18, linestyle = '--', color='k', linewidth=1.2)
plt.vlines(x=25918.921, ymin=0, ymax=46.18, linestyle = '--', color='k', linewidth=1.2)
plt.grid()
plt.legend()
# -






