from src.common.config import *
import xmltodict


def process_file(fname):
	"""Function to process every .xml file except XXXTransiti.xml and XXXLimitiTransito.xml,
	for which the function process_transit_file() is used.

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
	dic = xmltodict.parse(data)["NewDataSet"]

	# Delete unuseful information of the xml file
	del dic["xs:schema"]

	# Everything is under a key...
	dic = dic[next(iter(dic))]
	
	# New list of 24 dictionaries, one per hour
	m_list = []
	date = dic[0]['Data']
	for hour in range(1, 25):
		m_list.append({'Data': date, 'Ora': '{:02d}'.format(hour)})

	# Fill up the new list m_list with reformatted keys and values
	for h in dic:
		key1 = (int(h['Ora'])) - 1  # hours go from 1 to 24, list indices start from 0

		# Start iterating from the 4th key (first 3 keys are 'Data', 'Mercato', 'Ora')
		for i in list(h.keys())[3:]:
			key2 = h['Mercato'] + '_' + i + suffix(fname[11:-4])
			m_list[key1][key2] = float(h[i].replace(',', '.'))

	return m_list


def process_transit_file(fname):
	"""Function to process XXXTransiti.xml and XXXLimitiTransito.xml files (XXX = {MGP, MI1, MI2, ...})
	
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
	dic = xmltodict.parse(data)["NewDataSet"]

	# Delete unuseful information of the xml file
	del dic["xs:schema"]

	# Everything is under a key...
	dic = dic[next(iter(dic))]
	
	# New list of 24 dictionaries, one per hour
	m_list = []
	date = dic[0]['Data']
	for hour in range(1, 25):
		m_list.append({'Data': date, 'Ora': '{:02d}'.format(hour)})

	# Fill up the new dictionary m_dict with reformatted keys and values
	for h in dic:
		key1 = (int(h['Ora'])) - 1  # hours go from 1 to 24, list indices start from 0

		# Start iterating from the 6th key (first 5 keys are 'Data', 'Mercato', 'Ora', 'Da', 'A')
		for i in list(h.keys())[5:]:
			key2 = h['Mercato'] + '_' + h['Da'] + '_' + h['A'] + '_' + i
			m_list[key1][key2] = float(h[i].replace(',', '.'))

	return m_list


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
	dic = xmltodict.parse(data, postprocessor=is_float)["NewDataSet"]

	# Delete unuseful information of the xml file
	del dic["xs:schema"]

	# TODO: decide useless features to drop

	# Everything is under a key...
	dic = dic[next(iter(dic))]

	return dic


def is_float(path, key, value):
	try:
		return key, float(value)
	except (ValueError, TypeError):
		return key, value


def suffix(i):
	"""Generate a suffix for fields which have the same name in more than one file.
	
	Parameters
	----------
	i : str
		input string to generate the suffix

	Returns
	-------
	str
		the proper suffix
	"""

	switcher = {'Fabbisogno': '_Fabbisogno',
				'Prezzi': '_Prezzo',
				'PrezziConvenzionali': '_PrezzoConv',
				'StimeFabbisogno': '_StimeFabb'}
	return switcher.get(i, '')
