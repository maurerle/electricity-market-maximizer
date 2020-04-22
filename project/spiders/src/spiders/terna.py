from datetime import datetime
from time import sleep
from pathlib import Path
from typing import Dict, List
from sys import dont_write_bytecode
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementNotInteractableException
from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.keys import Keys
from src.common.config import DOWNLOAD, QUEUE, TERNA2, TERNA_LIMIT
import requests as req
from bs4 import BeautifulSoup as bs

dont_write_bytecode = True
class TernaReserve():
    """Reach the Terna's secondary reserve page and retrieve the daily data
    and the ones from the 01/02/2017 to create the database.
    When the file is downloaded, it is added to the processing queue.

    Attributes
    ----------
    driver : selenium.webdriver.firefox.webdriver.WebDriver
        web driver emulating the user's actions on the browser

    Methods
    -------
    getDaily()
    getHistory()
    download(href, f_cnt)
    
    Returns
    -------
    None
        exit from the while loops
    """
    def __init__ (self):
        self.driver = webdriver.Firefox(
            log_path='logs/geckodrivers.log'
        )
        self.driver.set_page_load_timeout(20)
        self.driver.get(TERNA2)
    
    def getDaily(self):
        """Find the link of Terna's secondary reserve data referred to the 
        current day.
        """
        today = datetime.now().strftime('%Y%m%d')

        # Wait until the table is not loaded
        while True:
            try:
                self.driver.find_element_by_class_name(
                    'terna-icon-download'
                )
                break
            except NoSuchElementException:
                sleep(1)
        # Find the download lnks
        soup = bs(self.driver.page_source, 'html.parser')
        for a in soup.find_all('a', href=True):
            if 'download.terna.it' and today in a['href']:
                self.download(a['href'], 0)
                break
                     
    def getHistory(self):
        """Find the link of all the Terna's secondary reserve data from the 
        starting day up to the day before the current one.
        
        Returns
        -------
        None
            exiting the while loops
        """
        # Day limits
        today = datetime.now().strftime('%Y%m%d')
        limit = TERNA_LIMIT.strftime('%Y%m%d')
        dayM = datetime.strptime(today, '%Y%m%d')
        daym = datetime.strptime(limit, '%Y%m%d')
        limit = (dayM - daym).days
        cnt = 2
        f_cnt = 0
        while True:
            # Wait until the table is not loaded
            while True:
                try:
                    self.driver.find_element_by_class_name(
                        'terna-icon-download'
                    )
                    break
                except NoSuchElementException:
                    sleep(1)

            soup = bs(self.driver.page_source, 'html.parser')
            # Find the download links
            for a in soup.find_all('a', href=True):
                if 'download.terna.it' in a['href'] and today not in a['href']:
                    if cnt == limit:
                        return None
                    cnt+=1
                    self.download(a['href'], f_cnt)
                    f_cnt+=1
            # Move to the next table
            nxt = self.driver.find_element_by_xpath(
                '/html/body/form/div[3]/div/div/div[1]/div/div/div[3]/div/'\
                'div/div/div/div/div/div[3]/div[1]/div/div/div/ul/li[3]/a'
            )
            nxt.click()
    
    def download(self, href, f_cnt):
        """Download the data by performing an HTTP get request and by saving
        the response into a file. Then add the file to the processing queue.
        
        Parameters
        ----------
        href : str
            download link
        f_cnt : int
            file counter for files with the same name
        """
        fname = href.split('/')[-1]
        fname = f'{f_cnt}_{fname}'
        if 'XLS' in fname:
            fname = fname.replace('XLS', 'xlsx')
        # Download files
        with open(f'{DOWNLOAD}/{fname}', 'wb') as file:
            res = req.get(href)
            file.write(res.content)
        # Add it to the queue for the processor
        QUEUE.put(fname)
