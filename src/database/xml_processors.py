from common.config import *
import xmltodict


def process_file(fname, PREFIX):
	with open(DOWNLOAD + '/' + fname, 'r') as file:
	    data = file.read()

	    # Convert the xml file to dictionary
	    dic = xmltodict.parse(data)["NewDataSet"]

	    # Delete unuseful information of the xml file
	    del dic["xs:schema"]

	    date = dic[next(iter(dic))][0]['Data']

	    # Create an empty dictionary with 24 keys (keys' format is yyyymmdd_hh)
	    mi_dict = {}
	    for hour in range(1,25):
	    	key = date + '_' + '{:02d}'.format(hour)
	    	mi_dict[key] = {}

	    # Fill up the new dictionary with reformatted keys and values
	    for h in dic[next(iter(dic))]:
	    	key1 = h['Data'] + '_' + '{:02d}'.format(int(h['Ora']))

	    	# Start iterating from the 4th key (first 3 keys are 'Data', 'Mercato', 'Ora')
	    	for i in list(h.keys())[3:]:
	    		key2 = h['Mercato'] + '_' + PREFIX + '_' + i
	    		mi_dict[key1][key2] = float(h[i].replace(',','.'))


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
	    mi_dict = {}
	    for hour in range(1,25):
	    	key = date + '_' + '{:02d}'.format(hour)
	    	mi_dict[key] = {}

	    # Fill up the new dictionary with reformatted keys and values
	    for h in dic[next(iter(dic))]:
	    	key1 = h['Data'] + '_' + '{:02d}'.format(int(h['Ora']))

	    	# Start iterating from the 6th key (first 5 keys are 'Data', 'Mercato', 'Ora', 'Da', 'A')
	    	for i in list(h.keys())[5:]:
	    		key2 = h['Mercato'] + '_' + h['Da'] + '_' + h['A'] + '_' + i
	    		mi_dict[key1][key2] = float(h[i].replace(',','.'))
