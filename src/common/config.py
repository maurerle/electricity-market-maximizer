import os
# ======================================
# LOGGER BOT
# ======================================
TOKEN = '1044047901:AAHWb7zhMQwWiiEg266rf3ZknIiMDQHlw0Q'
CHAT_IDS = [523755114, 166462336, 192294736, 396732122]


# ======================================
# SPIDERS
# ======================================

# Hide the Firefox window when automating with selenium
os.environ['MOZ_HEADLESS'] = '1'

# GME urls
DOWNLOAD = os.getcwd()+'/downloads'
RESTRICTION = 'https://www.mercatoelettrico.org/It/Download/DownloadDati.aspx'
GME_NEXT = [
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
		'fname':'MGP_Liquidita',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MGP_Liquidita'
	},
	{
		'fname':'MGP_Quantita',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MGP_Quantita'
	},
	{
		'fname':'MI1_LimitiTransito',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI1_LimitiTransito'
	},
	{
		'fname':'MI1_Prezzi',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI1_Prezzi'
	},
	{
		'fname':'MI1_PrezziConvenzionali',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI1_PrezziConvenzionali'
	},
	{
		'fname':'MI1_Quantita',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI1_Quantita'
	},
	{
		'fname':'MI2_LimitiTransito',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI2_LimitiTransito'
	},
	{
		'fname':'MI2_Prezzi',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI2_Prezzi'
	},
	{
		'fname':'MI2_PrezziConvenzionali',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI2_PrezziConvenzionali'
	},
	{
		'fname':'MI2_Quantita',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI2_Quantita'
	}
]
GME = [
	{
		'fname':'MI3_LimitiTransito',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI3_LimitiTransito'
	},
	{
		'fname':'MI3_Prezzi',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI3_Prezzi'
	},
	{
		'fname':'MI3_PrezziConvenzionali',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI3_PrezziConvenzionali'
	},
	{
		'fname':'MI3_Quantita',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI3_Quantita'
	},
	{
		'fname':'MI4_LimitiTransito',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI4_LimitiTransito'
	},
	{
		'fname':'MI4_Prezzi',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI4_Prezzi'
	},
	{
		'fname':'MI4_PrezziConvenzionali',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI4_PrezziConvenzionali'
	},
	{
		'fname':'MI4_Quantita',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI4_Quantita'
	},
	{
		'fname':'MI5_LimitiTransito',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI5_LimitiTransito'
	},
	{
		'fname':'MI5_Prezzi',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI5_Prezzi'
	},
	{
		'fname':'MI5_PrezziConvenzionali',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI5_PrezziConvenzionali'
	},
	{
		'fname':'MI5_Quantita',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI5_Quantita'
	},
	{
		'fname':'MI6_LimitiTransito',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI6_LimitiTransito'
	},
	{
		'fname':'MI6_Prezzi',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI6_Prezzi'
	},
	{
		'fname':'MI6_PrezziConvenzionali',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI6_PrezziConvenzionali'
	},
	{
		'fname':'MI6_Quantita',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI6_Quantita'
	},
	{
		'fname':'MI7_LimitiTransito',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI7_LimitiTransito'
	},
	{
		'fname':'MI7_Prezzi',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI7_Prezzi'
	},
	{
		'fname':'MI7_PrezziConvenzionali',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI7_PrezziConvenzionali'
	},
	{
		'fname':'MI7_Quantita',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MI7_Quantita'
	},
	{
		'fname':'MSD_ServiziDispacciamento',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MSD_ServiziDispacciamento'
	},
	{
		'fname':'MB_PTotali',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MB_PTotali'
	},
	{
		'fname':'MB_PRiservaSecondaria',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MB_PRiservaSecondaria'
	},
	{
		'fname':'MB_PAltriServizi',
		'url':'https://www.mercatoelettrico.org/It/download/DownloadDati.aspx?val=MB_PAltriServizi'
	}	
]

# Dynamic file history
HISTORY=[]


# ======================================
# DATABASE
# ======================================

# To be changed with Polito cluster credentials
MONGO_STRING = "mongodb+srv://new-user:politomerda@cluster0-awyti.mongodb.net/test?retryWrites=true&w=majority"

DB_NAME = "InterProj"