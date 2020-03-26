from src.common.config import DOWNLOAD
import xmltodict
from datetime import datetime
import pandas as pd
import numpy as np
from src.common.config import MI

def process_OffPub(fname):
	"""Function to process XXXOffertePubblice.xml files (XXX = {MGP, MI1, MI2, ...})
	
	Parameters
	----------
	fname : str
		name of the .xml file
	
	Returns
	-------
	m_list : list
		list of dictionaries with the reformatted data
	"""

	with open(DOWNLOAD + '/' + fname, 'r') as file:
		data = file.read()

	# Convert the xml file to dictionary
	dic = xmltodict.parse(data, postprocessor=type_conv)["NewDataSet"]

	# Delete unuseful information of the xml file
	del dic["xs:schema"]

	dic = dic[next(iter(dic))]
	for i in dic:
		del i['UNIT_REFERENCE_NO']
		del i['TYPE_CD']
		del i['TRANSACTION_REFERENCE_NO']
		del i['AWARDED_QUANTITY_NO']
		del i['MERIT_ORDER_NO']
		del i['PARTIAL_QTY_ACCEPTED_IN']
		del i['ADJ_QUANTITY_NO']
		del i['GRID_SUPPLY_POINT_NO']
		del i['ZONE_CD']
		del i['SUBMITTED_DT']
		del i['BILATERAL_IN']

		try:
			del i['BALANCED_REFERENCE_NO']
		except:
			pass
	
	if 'MI' in i['MARKET_CD']:
		idx = i['BID_OFFER_DATE_DT']
		if idx not in MI:
			MI[idx] = {
				'dem':[],
				'sup':[]
			}

		d, s = dataPool(dic, 'MI')
		MI[idx]['dem'].append(d)
		MI[idx]['sup'].append(s)
		
		if len(MI[idx]['dem']) == 7:
			dQ = pd.DataFrame()
			dP = pd.DataFrame()
			sQ = pd.DataFrame()
			sP = pd.DataFrame()
			for i in range(7):
				dQ = pd.concat((dQ, MI[idx]['dem'][i].Q), axis=1, sort=False).sum(axis=1)
				dP = pd.concat((dP, MI[idx]['dem'][i].P), axis=1, sort=False).mean(axis=1)
				sQ = pd.concat((sQ, MI[idx]['sup'][i].Q), axis=1, sort=False).sum(axis=1)
				sP = pd.concat((sP, MI[idx]['sup'][i].P), axis=1, sort=False).mean(axis=1)
			MI.pop(idx)
			dem = pd.DataFrame({
				'P':dP,
				'Q':dQ,
			})
			sup = pd.DataFrame({
				'P':sP,
				'Q':sQ,
			})
			dem['MARKET'] = 'MI'
			sup['MARKET'] = 'MI'
			dem['DATE'] = idx
			sup['DATE'] = idx

			return dem, sup		
		else:
			return -1, -1
	elif 'MGP' in i['MARKET_CD']:	
		return dataPool(dic, 'MI/MGP')
	elif 'MSD' in i['MARKET_CD']:
		return dataPool(dic, 'MSD')


def getCurve(df):
	curve = pd.DataFrame(columns=['OPS','P','Q'])
	cnt = 0
	for op in df['OPERATORE'].unique():
		new = pd.DataFrame(columns=['OPS','P','Q'])
		temp = df.where(df['OPERATORE']==op).dropna()
		new.loc[cnt] = [
			op,
			np.mean(temp['ENERGY_PRICE_NO']),
			np.sum(temp['QUANTITY_NO'])
		]
		cnt+=1
		curve = pd.concat([curve, new], axis= 0)
	
	curve = curve.set_index('OPS')
	curve['MARKET'] = df.iloc[0]['MARKET_CD']
	curve['DATE'] = df.iloc[0]['BID_OFFER_DATE_DT']
	
	
	return curve


def type_conv(path, key, value):
	if key == 'BID_OFFER_DATE_DT' or key == 'SUBMITTED_DT':
		return key, value
	if key == 'BILATERAL_IN':
		return key, bool(value)
	try:
		return key, float(value)
	except (ValueError, TypeError):
		return key, value


def dataPool(converted, m):
	
	data = pd.DataFrame(converted)
	data = (
		data
		.where(data['OPERATORE']!='Bilateralista')
		.where(data['STATUS_CD'].isin(['ACC', 'REJ']))
		.dropna()
	)
	if m == 'MSD':
		data = (
			data
			.where(data['SCOPE'].isin(['GR1', 'RS']))
			.dropna()
		)

	off = (
		data
		.where(data['PURPOSE_CD']=='OFF')
		.drop(columns='PURPOSE_CD')
		.dropna()
	)
	
	bid = (
		data
		.where(data['PURPOSE_CD']=='BID')
		.drop(columns='PURPOSE_CD')
		.dropna()
	)
	
	dem = getCurve(bid)
	sup = getCurve(off)
	
	return dem, sup