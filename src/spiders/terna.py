from datetime import datetime
import time
from pathlib import Path
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
#from src.common.config import *

import logging
import logging.config

# logging.config.fileConfig('src/logging.conf')
# logger = logging.getLogger(__name__)
path=(Path.cwd())
path = str(path.parents[1] / 'downloads')

class TernaSpider():
    def __init__(self):
        profile = webdriver.FirefoxProfile()  # path -- gekodriver
        profile.set_preference("browser.download.folderList", 2)
        profile.set_preference("browser.helperApps.alwaysAsk.force", False)
        profile.set_preference("browser.download.manager.showWhenStarting",False)
        profile.set_preference("browser.download.dir", path)
        profile.set_preference("browser.download.downloadDir", path)
        profile.set_preference("browser.download.defaultFolder", path)
        profile.set_preference(
        	"browser.helperApps.neverAsk.saveToDisk", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, application/-csv"
    	)

        self.driver = webdriver.Firefox(profile, service_log_path='../../logs/geckodrivers.log')
        self.driver.set_page_load_timeout(20)
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
                graph = self.driver.find_element_by_css_selector("visual-container-modern.visual-container-component:nth-child("+mod+") > transform:nth-child(1)")
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
        time.sleep(6)
        self.driver.close()

    def checkdownload(self, file_path):
        while not os.path.exists(file_path):

            time.sleep(1)

        if os.path.isfile(file_path):

            print ("File is downloaded in following dir: " + str(file_path))
        else:
            raise ValueError("%s isn't a file!" % file_path)

    def setFname(self, pat, fname, date):
        time.sleep(5)
        p = Path(pat)
        flist = [x for x in p.glob('data.xlsx')]
        print(flist)
        p_file_0 = flist[0]
        print(type(p_file_0))
        print(p_file_0.name)
        target = Path(str(pat) +'/'+ str(fname) + str(date) + str('.xlsx'))
        target.exists()
        p_file_0.replace(target)

    def getHistory(self, url, frm, mod, d):

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
        historyowned = False

        while not historyowned:

            try:
                # click on menu item Transmission
                # click on menu item Scheduled Foreign Echange
                # set Country
                # set month
                # set Year
                # Click on Options button
                btn = self.driver.find_element_by_class_name('vcMenuBtn')
                self.driver.execute_script('arguments[0].click();', btn)
                self.action.move_to_element(btn).perform()
                # Click on Export Data
                btn = self.driver.find_element_by_xpath(
                    "/html/body/div[" + d + "]/drop-down-list/ng-transclude/ng-repeat[1]/drop-down-list-item/ng-transclude/ng-switch/div")
                self.action.move_to_element(btn).perform()
                self.driver.execute_script('arguments[0].click();', btn)
                # click on Export Data
                # click on download button


                historyowned=True

            except NoSuchElementException:
                print('Try Again')
                time.sleep(1)



load = [
            { #Total Load
                'name': 'TotalLoad',
                'url': 'https://www.terna.it/it/sistema-elettrico/transparency-report/total-load',
                'frame': 'iframeTotalLoad',
                'ntch': '6',
                'div': '9'

            },
            {#Market Load
                'name':'MarketLoad',
                'url': 'https://www.terna.it/it/sistema-elettrico/transparency-report/market-load',
                'frame': 'iframeMarketLoad',
                'ntch': '6',
                'div':'9'

            },
            {#peak-valley-load
                'name':'PeakValleyLoad',
                'url': 'https://www.terna.it/it/sistema-elettrico/transparency-report/peak-valley-load',
                'frame': 'iframePeakValleyLoad',
                'ntch': '23',
                'div':'9'

            }
        ]
generation = [
            {
                'name': 'ActualGen',
                'url': 'https://www.terna.it/it/sistema-elettrico/transparency-report/actual-generation',
                'frame': 'iframeActualGen',
                'ntch': '28',
                'div': '9'

            },
            {
                'name':'RenewableGen',
                'url': 'https://www.terna.it/it/sistema-elettrico/transparency-report/renewable-generation',
                'frame': 'iframeRenewableGen',
                'ntch': '13',
                'div':'9'

            },
            {
                'name':'EnergyBal',
                'url': 'https://www.terna.it/it/sistema-elettrico/transparency-report/energy-balance',
                'frame': 'iframeEnergyBal',
                'ntch': '31',
                'div':'9'

            },
            {
                'name':'InstalledCap',
                'url': 'https://www.terna.it/it/sistema-elettrico/transparency-report/installed-capacity',
                'frame': 'iframeInstalledCap',
                'ntch': '8',
                'div':'9'

            }
        ]
dl_center = {
    'url':'https://www.terna.it/it/sistema-elettrico/transparency-report/download-center',
    'frame': 'iframeDownload',
    'ntch': '',
    'div':'div'
}



###############################################################################
xpath=Path.cwd()
print('cwd ' + str(xpath))
print('[-1] ' + str(xpath.parents[1]))
xpath = xpath.parents[1] / 'downloads/'
xpath_file = xpath / 'data.xlsx'
print(xpath)
day = datetime.now().strftime('%d%m%Y')
print(day)

for item in load:
    print(item['frame'])
    spider = TernaSpider()
    spider.getData(item['url'], item['frame'], item['ntch'], item['div'])
    spider.checkdownload(xpath_file)
    spider.setFname(xpath, item['name'], day)
    spider.quit()


for gen in generation:
    print(gen['frame'])
    spider = TernaSpider()
    spider.getData(gen['url'], gen['frame'], gen['ntch'], gen['div'])
    spider.checkdownload(xpath_file)
    spider.setFname(xpath, gen['name'], day)
    spider.quit()




'''

TODO: 
getHistory function (download center)
introdurre cicli for in getData
check installedCap downlaod

'''
