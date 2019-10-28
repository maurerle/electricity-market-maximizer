import os

# ======================================
# SPIDERS
# ======================================

# Hide the Firefox window when automating with selenium
os.environ['MOZ_HEADLESS'] = '1'

# GME urls
DOWNLOAD = '/home/luca/Codes/smartgrids/downloads'
RESTRICTION = 'https://www.mercatoelettrico.org/It/Download/DownloadDati.aspx'
GME = [
	{
		'fname':'MGP_PrezziConvenzionali',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MGP_PrezziConvenzionali'
	},
	{
		'fname':'MGP_LimitiTransito',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MGP_LimitiTransito'
	},
	{
		'fname':'MGP_StimeFabbisogno',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MGP_StimeFabbisogno'
	},
	{
		'fname':'MGP_Prezzi',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MGP_Prezzi'
	},
	{
		'fname':'MGP_OfferteIntegrativeGrtn',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MGP_OfferteIntegrativeGrtn'
	},
	{
		'fname':'MGP_Fabbisogno',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MGP_Fabbisogno'
	},
	{
		'fname':'MGP_Transiti',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MGP_Transiti'
	},
	{
		'fname':'MGP_Liquitidtà',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MGP_Liquitidtà'
	},
	{
		'fname':'MGP_Quantità',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MGP_Quantità'
	}
]

# Dynamic file history
HISTORY=[]
