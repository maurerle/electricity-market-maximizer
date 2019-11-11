import time
import os
from typing import Dict, List

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait as wait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from src.common.config import *

import logging
import logging.config

# logging.config.fileConfig('src/logging.conf')
# logger = logging.getLogger(__name__)

class TernaSpider():
    def __init__(self):
        profile = webdriver.FirefoxProfile()  # path -- gekodriver
        profile.set_preference("browser.download.folderList", 2)
        profile.set_preference("browser.helperApps.alwaysAsk.force", False)
        profile.set_preference("browser.download.manager.showWhenStarting",False)
        profile.set_preference("browser.download.dir", DOWNLOAD)
        profile.set_preference("browser.download.downloadDir", DOWNLOAD)
        profile.set_preference("browser.download.defaultFolder", DOWNLOAD)
        profile.set_preference(
        	"browser.helperApps.neverAsk.saveToDisk", 
        	"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    	)

        self.driver = webdriver.Firefox(profile, log_path='../../logs/geckodrivers.log')

        self.driver = webdriver.Firefox(profile)
        self.driver.set_page_load_timeout(15)
        self.action = ActionChains(self.driver)


    def getData(self,url, frm, mod, d):
        self.driver.get(url)
        self.driver.switch_to.frame(self.driver.find_element_by_id(frm))

        try:
            wait(self.driver, 1).until(ec.frame_to_be_available_and_switch_to_it((
            	By.XPATH, 
            	'/html/body/div/iframe'
        	)))
            print("trying")
        except TimeoutException:
            time.sleep(1)
            print("again")
            
        terna_owned = False
        while not terna_owned:
            try:       
                # Click on graph
                graph = self.driver.find_element_by_css_selector("visual-container-modern.visual-container-component:nth-child("+mod+''") > transform:nth-child(1)")
                self.driver.execute_script('arguments[0].click();', graph)
                self.action.move_to_element(graph).perform()
                # Click on Options button
                btn = self.driver.find_element_by_class_name('vcMenuBtn')
                self.driver.execute_script('arguments[0].click();', btn)
                self.action.move_to_element(btn).perform()
                # Click on Export Data
                btn = self.driver.find_element_by_xpath("/html/body/div["+d+"]/drop-down-list/ng-transclude/ng-repeat[1]/drop-down-list-item/ng-transclude/ng-switch/div")
                self.action.move_to_element(btn).perform()
                self.driver.execute_script('arguments[0].click();', btn)
                terna_owned = True



            except NoSuchElementException:
                print('Try Again')
                time.sleep(1)

        terna_owned = False
        while not terna_owned:
            try:
                # Download button
                btn = self.driver.find_element_by_class_name("primary") 
                self.driver.execute_script('arguments[0].click();', btn)
                terna_owned = True       
            except NoSuchElementException:
                print('Try Again')
                time.sleep(1)

    def quit(self):
        self.driver.close()


#URL = "https://www.terna.it/it/sistema-elettrico/transparency-report/actual-generation"

load = [
            { #Total Load
                'url':'https://www.terna.it/it/sistema-elettrico/transparency-report/total-load',
                'frame' : 'iframeTotalLoad',
                'ntch' : '6',
                'div':'9'

            },
            {#Market Load
                'url': 'https://www.terna.it/it/sistema-elettrico/transparency-report/market-load',
                'frame': 'iframeMarketLoad',
                'ntch': '6',
                'div':'8'

            },
            {#peak-valley-load
                'url': 'https://www.terna.it/it/sistema-elettrico/transparency-report/peak-valley-load',
                'frame': 'iframePeakValleyLoad',
                'ntch': '23',
                'div':'8'


            }
        ]
print(DOWNLOAD)
for item in load:
    print(item['frame'])
    spider = TernaSpider()
    spider.getData(item['url'], item['frame'], item['ntch'], item['div'])
    spider.quit()

'''
TODO: 
1 Function ---> che mi prende giorno per giorno per ogni categoria i dati e generalizzala
2 Function ---> periodo start and end per tutte le categorie della dashboard
3 Function ---> che si genera il filename con la data e lo rinomina 
4 Function ---> checkdownload 


'''
