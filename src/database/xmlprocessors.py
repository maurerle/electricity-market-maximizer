from common.config import *
import xmltodict


def process_file(fname):
	with open(DOWNLOAD + '/' + fname, 'r') as file:
	    data = file.read()

	    # Convert the xml file to dictionary
	    dic = xmltodict.parse(data)["NewDataSet"]

	    # Delete unuseful information of the xml file
	    del dic["xs:schema"]

	    date = dic[next(iter(dic))][0]['Data']

	    # Create an empty dictionary with 24 keys (keys' format is yyyymmdd_hh)
	    m_dict = {}
	    for hour in range(1,25):
	    	key = date + '_' + '{:02d}'.format(hour)
	    	m_dict[key] = {'Data': date, 'Ora': '{:02d}'.format(hour)}

	    # Fill up the new dictionary m_dict with reformatted keys and values
	    for h in dic[next(iter(dic))]:
	    	key1 = h['Data'] + '_' + '{:02d}'.format(int(h['Ora']))

	    	# Start iterating from the 4th key (first 3 keys are 'Data', 'Mercato', 'Ora')
	    	for i in list(h.keys())[3:]:
	    		key2 = h['Mercato'] + '_' + i + suffix(fname[11:-4])
	    		m_dict[key1][key2] = float(h[i].replace(',','.'))

	return m_dict


def process_transit_file(fname):
	"""
	To process XXXTransiti.xml and XXXLimitiTransito.xml files (XXX = {MGP, MI1, MI2, ...})
	"""

	with open(DOWNLOAD + '/' + fname, 'r') as file:
	    data = file.read()

	    # Convert the xml file to dictionary
	    dic = xmltodict.parse(data)["NewDataSet"]

	    # Delete unuseful information of the xml file
	    del dic["xs:schema"]

	    date = dic[next(iter(dic))][0]['Data']

	    # Create an empty dictionary with 24 keys (keys' format is yyyymmdd_hh)
	    m_dict = {}
	    for hour in range(1,25):
	    	key = date + '_' + '{:02d}'.format(hour)
	    	m_dict[key] = {'Data': date, 'Ora': '{:02d}'.format(hour)}

	    # Fill up the new dictionary m_dict with reformatted keys and values
	    for h in dic[next(iter(dic))]:
	    	key1 = h['Data'] + '_' + '{:02d}'.format(int(h['Ora']))

	    	# Start iterating from the 6th key (first 5 keys are 'Data', 'Mercato', 'Ora', 'Da', 'A')
	    	for i in list(h.keys())[5:]:
	    		key2 = h['Mercato'] + '_' + h['Da'] + '_' + h['A'] + '_' + i
	    		m_dict[key1][key2] = float(h[i].replace(',','.'))

	return m_dict

def suffix(i):
	switcher = {'Fabbisogno':'_Fabbisogno',
	            'Prezzi':'_Prezzo', 
	            'PrezziConvenzionali':'_PrezzoConv',
	            'StimeFabbisogno':'_StimeFabb'}
	return switcher.get(i,'')