from datetime import datetime
from time import sleep
from pathlib import Path
from typing import Dict, List
from sys import dont_write_bytecode
from src.loggerbot.bot import bot
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
class TernaSpider():
    """Reach the Energy Balance, Ttoal and Market load fields of the Terna 
    website and download datavreferred to an indicated range of days. Then 
    store their name in a queue from which they will be processed.
    
    Parameters
    ----------
    logger : logging.logger
        logger instance to display and save logs
    
    Attributes
    ----------
    driver : selenium.webdriver.firefox.webdriver.WebDriver
        web driver emulating the user's actions on the browser
    action : selenium.webdriver.common.action_chains.ActionChains
        used to perform mouse hovering
    passed : bool
        flag to notify that a connection or congestion error occurred
    
    Methods
    -------
    getData(url, start, end)
    setFname(date)    
    """
    def __init__(self, logger):
        profile = webdriver.FirefoxProfile()
        profile.set_preference("browser.download.folderList", 2)
        profile.set_preference("browser.helperApps.alwaysAsk.force", False)
        profile.set_preference(
            "browser.download.manager.showWhenStarting",
            False
        )
        profile.set_preference("browser.download.dir", DOWNLOAD)
        profile.set_preference("browser.download.downloadDir", DOWNLOAD)
        profile.set_preference("browser.download.defaultFolder", DOWNLOAD)
        profile.set_preference(
        	"browser.helperApps.neverAsk.saveToDisk", 
            "application/vnd.openxmlformats-officedocument.spreadsheetml."\
            "sheet, application/-csv"
    	)

        self.driver = webdriver.Firefox(
            profile, 
            log_path='logs/geckodrivers.log'
        )
        self.driver.set_page_load_timeout(20)
        self.action = ActionChains(self.driver)
        self.log = logger
        self.passed = False

    def getData(self, item, start, end):
        """The spider get the url, find the "Custom Range" button and insert 
        the starting and ending download period. Then moves back to the graph,
        enable the Options button, select the "Export data" entry of the 
        dropdown menu and finally downloads data.
        If any connection or congestion problem occurs, the process is 
        automatically repeated.
        
        Parameters
        ----------
        item : dict
            download information of the Terna item
        start : str
            starting download period
        end : str
            ending download period
        """
        self.driver.get(item['url'])
        self.driver.switch_to.frame(
            self.driver.find_element_by_id(item['iframe'])
        )

        while True:
            try:
                wait(self.driver, 1).until(
                    ec.frame_to_be_available_and_switch_to_it((
                        By.XPATH, 
                        '/html/body/div/iframe'
                    ))
                )
                break
            except TimeoutException:
                sleep(1)
            
        while True:
            try: 
                # Get the parent div
                parent = self.driver.find_element_by_class_name("canvasFlexBox")   
                # Div
                
                btn = parent.find_element_by_css_selector(
                    "visual-container-modern.visual-container-component:nth"\
                    f"-child({item['child']}) > transform:nth-child(1) > div:nth-child(1) "\
                    "> div:nth-child(3) > visual-modern:nth-child(1) "\
                    "> div:nth-child(1)"
                )
                self.driver.execute_script('arguments[0].click();', btn)
                self.log.info('[TERNA] Frame switched.')
                while True:
                    try:
                        wait(parent, 1).until(
                            ec.visibility_of_element_located((
                                By.TAG_NAME, 
                                'input'
                            ))
                        )
                        break
                    except TimeoutException:
                        sleep(1)
                
                #Inputs
                form = parent.find_elements_by_tag_name("input")
                self.driver.execute_script('arguments[0].click();', form[0])
                sleep(1)
                form[0].send_keys(start)
                self.driver.execute_script('arguments[0].click();', form[1])
                sleep(1)
                form[1].send_keys(end)
                self.log.info('[TERNA] Date inserted')
                
                break

            except NoSuchElementException:
                sleep(1)
            
            except StaleElementReferenceException:
                self.log.error('[TERNA] Stale error. Try again..')
                
                try:
                    bot('ERROR', 'TERNA', 'Stale error. Try again..')
                except:
                    pass
                
                self.getData(item, start, end)
                
                break
            
            except ElementNotInteractableException:
                self.log.error('[TERNA] Interactable error. Try again..')
                
                try:
                    bot('ERROR', 'TERNA', 'Interactable error. Try again..')
                except:
                    pass
                
                self.getData(item, start, end)
                
                break

        while True:
            try: 
                # Graph
                graph = self.driver.find_element_by_css_selector(
                    "#pvExplorationHost > div > div > exploration > div "\
                    "> explore-canvas-modern > div > div.canvasFlexBox > div "\
                    "> div.displayArea.disableAnimations.fitToPage "\
                    "> div.visualContainerHost > visual-container-repeat "\
                    f"> visual-container-modern:nth-child({item['graph']}) > transform"
                )
                self.driver.execute_script('arguments[0].click();', graph)
                self.action.move_to_element(graph).perform()
                
                # Options
                btn = parent.find_element_by_class_name('vcMenuBtn')
                self.driver.execute_script('arguments[0].click();', btn)
                self.action.move_to_element(btn).perform()
                self.log.info('[TERNA] Options Clicked.')

                # Export Data
                btn = parent.find_element_by_xpath(
                    "/html/body/div[8]/drop-down-list/ng-transclude/ng-repeat[1]/drop-down-list-item/ng-transclude/ng-switch/div"
                )
                self.action.move_to_element(btn).perform()
                self.driver.execute_script('arguments[0].click();', btn)
                break

            except NoSuchElementException:
                sleep(1)
            
            except StaleElementReferenceException:
                self.log.error('[TERNA] Stale error. Try again..')
                
                try:
                    bot('ERROR', 'TERNA', 'Stale error. Try again..')
                except:
                    pass
                
                self.getData(item, start, end)
                
                break
            
            except ElementNotInteractableException:
                self.log.error('[TERNA] Interactable error. Try again..')
                
                try:
                    bot('ERROR', 'TERNA', 'Interactable error. Try again..')
                except:
                    pass
                
                self.getData(item, start, end)
                
                break



        if not self.passed:
            while True:
                try:
                    # Download button
                    btn = self.driver.find_element_by_class_name("primary") 
                    self.driver.execute_script('arguments[0].click();', btn)
                    self.log.info('[TERNA] Export data.')
                    
                    break       
                except NoSuchElementException:
                    sleep(1)
            
            self.passed = True    
            self.setFname(item, start)

    def setFname(self, item, date):
        """When the .xlsx data is correctly downloaded its name is changed into
        EnergyBalDDMMYYY, TotalLoadDDMMYYY or MarketLoadDDMMYYY
        
        Parameters
        ----------
        item : dict
            download information of the Terna item
        date : str
            date data are referred to
        
        Returns
        -------
        None
            Only to correctly exit the two loops
        """
        date = date.replace('/','')
        while True:
            for files in Path(DOWNLOAD).iterdir():
                if 'data.xlsx' in str(files):
                    target = Path(f"{DOWNLOAD}/{item['name']}{date}.xlsx")
                    self.log.info(f"[TERNA] {item['name']}{date}.xlsx"\
                        " downloaded."
                    )
                    sleep(2)
                    files.replace(target)
                    
                    QUEUE.put(f"{item['name']}{date}.xlsx")
                    
                    return None
        sleep(1)


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
    download(href)
    
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
                self.download(a['href'])
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
                    self.download(a['href'])
            # Move to the next table
            nxt = self.driver.find_element_by_xpath(
                '/html/body/form/div[3]/div/div/div[1]/div/div/div[3]/div/'\
                'div/div/div/div/div/div[3]/div[1]/div/div/div/ul/li[3]/a'
            )
            nxt.click()
    
    def download(self, href):
        """Download the data by performing an HTTP get request and by saving
        the response into a file. Then add the file to the processing queue.
        
        Parameters
        ----------
        href : str
            download link
        """
        fname = href.split('/')[-1]
        # Download files
        with open(f'{DOWNLOAD}/{fname}', 'wb') as file:
            res = req.get(href)
            file.write(res.content)
        # Add it to the queue for the processor
        QUEUE.put(fname)