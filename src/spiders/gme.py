import time
import requests as req
from common.config import *
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

class GMESpider():
	def __init__(self, log):
		# Set the firefox profile preferences
		profile = webdriver.FirefoxProfile()
		profile.set_preference("browser.download.folderList", 2)
		profile.set_preference("browser.helperApps.alwaysAsk.force", False)
		profile.set_preference("browser.download.manager.showWhenStarting",False)
		profile.set_preference("browser.download.dir", DOWNLOAD)
		profile.set_preference("browser.download.downloadDir", DOWNLOAD)
		profile.set_preference("browser.download.defaultFolder", DOWNLOAD)
		profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "text/html")
		
		# Class init
		self.driver = webdriver.Firefox(profile)
		self.log = log
		
		self.passRestrictions()
	
		
	def passRestrictions(self):
		self.driver.get(RESTRICTION)
		# Flag the Agreement checkboxes
		_input = self.driver.find_element_by_id('ContentPlaceHolder1_CBAccetto1')
		_input.click()
		_input = self.driver.find_element_by_id('ContentPlaceHolder1_CBAccetto2')
		_input.click()
		# Submit the agreements
		_input = self.driver.find_element_by_id('ContentPlaceHolder1_Button1')
		_input.click()
		self.log.info("Agreements passed")
	
		
	def getData(self, gme, start, end):
		self.log.info("Retrieving data from {} to {}".format(start, end))
		self.driver.get(gme['url'])
		# Set the starting and endig date.
		# The GME has the one-month restriction
		_input = self.driver.find_element_by_id('ContentPlaceHolder1_tbDataStart')
		_input.send_keys(start)
		_input = self.driver.find_element_by_id('ContentPlaceHolder1_tbDataStop')
		_input.send_keys(end)
		
		# Download file
		_input = self.driver.find_element_by_id('ContentPlaceHolder1_btnScarica')
		_input.click()
		
		# Check if download succeded
		self.checkDownload(
			self.getFname(gme['fname'], start, end)
		)
	
		
	def getFname(self, fname, start, end):
		dds,mms,yys = start.split("/")
		dde,mme,yye = end.split("/")
		period = yys+mms+dds+yye+mme+dde
		fname += period
		fname += '.zip'
		
		return fname
	
	
	def checkDownload(self, fname):
		if os.path.isfile(DOWNLOAD+'/'+fname):
			self.log.info("{} file downloaded".format(fname))
		else:
			self.log.warning("{} download failed".format(fname))
