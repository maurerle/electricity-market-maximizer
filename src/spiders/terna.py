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


    def getData(self,url):
        self.driver.get(url)
        self.driver.switch_to.frame(self.driver.find_element_by_id("iframeEnergyBal"))

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
                parent = self.driver.find_element_by_class_name("canvasFlexBox")   
                # Div
                btn = parent.find_element_by_css_selector("visual-container-modern.visual-container-component:nth-child(34) > transform:nth-child(1) > div:nth-child(1) > div:nth-child(3) > visual-modern:nth-child(1) > div:nth-child(1)")
                print(f'customRange:{btn}')
                self.driver.execute_script('arguments[0].click();', btn)
                time.sleep(10)
                
                #Inputs
                form = parent.find_elements_by_tag_name("input")
                self.driver.execute_script('arguments[0].click();', form[0])
                time.sleep(1)
                form[0].send_keys('15/11/2019')
                time.sleep(2)
                self.driver.execute_script('arguments[0].click();', form[1])
                time.sleep(1)
                form[1].send_keys('15/11/2019')
                parent.click()
                # Graph
                graph = parent.find_element_by_css_selector("visual-container-modern.visual-container-component:nth-child(31) > transform:nth-child(1)")
                #graph = self.driver.find_element_by_css_selector("visual-container-modern.visual-container-component:nth-child(31) > transform:nth-child(1)")
                print(f'graph:{graph}')
                self.driver.execute_script('arguments[0].click();', graph)
                self.action.move_to_element(graph).perform()
                
                # Options
                btn = parent.find_element_by_class_name('vcMenuBtn')
                print(f'Menu:{btn}')
                self.driver.execute_script('arguments[0].click();', btn)
                self.action.move_to_element(btn).perform()
                
                # Export Data
                btn = parent.find_element_by_xpath("/html/body/div[9]/drop-down-list/ng-transclude/ng-repeat[1]/drop-down-list-item/ng-transclude/ng-switch/div")
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

    def getHistory(self, url, frm):

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
        
        try:
            wait(self.driver, 1).until(ec.frame_to_be_available_and_switch_to_it((
                By.XPATH,
                '/html/body/div/iframe'
            )))
            print("trying")
        except TimeoutException:
            time.sleep(1)
            print("again")
        gparent = self.driver.find_element_by_tag_name('html')
        historyowned = False
        
        while not historyowned: 
            try:
                # click on menu item Transmission
                print('clicking on transmission...')
                wait(self.driver, 1).until(ec.element_to_be_clickable((
            	By.CSS_SELECTOR, 
            	"visual-container-modern.visual-container-component:nth-child(4) > transform:nth-child(1)"
        	    )))
                parent = self.driver.find_element_by_css_selector(".visualContainerHost > visual-container-repeat:nth-child(1)")
                temp = parent.find_element_by_css_selector("visual-container-modern.visual-container-component:nth-child(14) > transform:nth-child(1)")
                temp.click()
                temp = parent.find_element_by_css_selector("visual-container-modern.visual-container-component:nth-child(19) > transform:nth-child(1)")
                temp.click()
                self.selectYear(parent, 2016)
                time.sleep(1)
                self.selectYear(parent, 2017)
                time.sleep(1)
                self.selectMonth(parent, gparent, 'Jan')
                temp = parent.find_element_by_css_selector("visual-container-modern.visual-container-component:nth-child(21) > transform:nth-child(1)")
                temp.click()
                self.selectYear(parent, 2016)
                time.sleep(1)
                self.selectYear(parent, 2017)
                time.sleep(1)
                self.selectMonth(parent, gparent, 'Jan')

                historyowned=True

            except NoSuchElementException:
                print('Try Again')
                time.sleep(1)
            except TimeoutException:
                print('Try Again Time')
                time.sleep(1)
        
        while not historyowned:
            try:
                # Download button
                btn = self.driver.find_element_by_class_name("primary")
                self.driver.execute_script('arguments[0].click();', btn)
                historyowned = True
            except NoSuchElementException:
                print('Try Again')
                time.sleep(1)

    def selectYear(self, elem, year):
        wait(elem, 1).until(ec.element_to_be_clickable((
            By.XPATH, 
            f"//*[text()={year}]"
        )))
        #ybar = driver.find_element_by_class_name("slicerItemsContainer")
        #ylist = driver.find_elements_by_class_name("individualItemContainer")
        item = elem.find_element_by_xpath(f"//*[text()={year}]")
        item.click()

    def selectMonth(self, elem, gelem, month):
        month = 'Feb'
        wait(elem, 2).until(ec.element_to_be_clickable((
            By.CSS_SELECTOR, 
            "visual-container-modern.visual-container-component:nth-child(16) > transform:nth-child(1) > div:nth-child(1) > div:nth-child(3) > visual-modern:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(3) > div:nth-child(1)"
        )))
        #item = elem.find_element_by_css_selector("visual-container-modern.visual-container-component:nth-child(16) > transform:nth-child(1) > div:nth-child(1) > div:nth-child(3) > visual-modern:nth-child(1) > div:nth-child(1) > div:nth-child(1) > div:nth-child(3) > div:nth-child(1)")
        #item.click()
        time.sleep(1)
        for i in range(1,10):
            item = gelem.find_element_by_css_selector(f"div.row:nth-child({i}) > div:nth-child(1)")
            self.driver.execute_script("arguments[0].click();", item)
            time.sleep(1)


#passare in config
TERNA_URL = {
                'name':'EnergyBal',
                'url': 'https://www.terna.it/it/sistema-elettrico/transparency-report/energy-balance',
                'frame': 'iframeEnergyBal',
                'ntch': '31',
                'div':'9'

            }
#start dp1574187165921
#end dp1574187165922
#custom range visual-container-modern.visual-container-component:nth-child(34) > transform:nth-child(1)

###############################################################################
"""
#just some uses of Path.
xpath=Path.cwd()
print('cwd ' + str(xpath))
print('[-1] ' + str(xpath.parents[1]))
xpath = xpath.parents[1] / 'downloads/'
xpath_file = xpath / 'data.xlsx'
print(xpath)
day = datetime.now().strftime('%d%m%Y')
print(day)

#Download Center
print(dl_center['frame'])
spider = TernaSpider()
spider.getHistory(dl_center['url'], dl_center['frame'])
#spider.quit()
"""
# #GENERATION ACQUISITION TODAY - all categories
spider = TernaSpider()
spider.getData(TERNA_URL['url'])
spider.checkdownload(xpath_file)
spider.setFname(xpath, TERNA_URL['name'], day)
spider.quit()




'''

TODO: 
getHistory function (download center)
introdurre cicli for in getData
check installedCap downlaod

'''
