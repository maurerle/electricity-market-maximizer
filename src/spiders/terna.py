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
from src.common.config import DOWNLOAD, TERNA, QUEUE

dont_write_bytecode = True
class TernaSpider():
    """Reach the Energy Balance field of the Terna website and download data
    referred to an indicated range of days. Then store their name in a queue 
    from which they will be processed.
    
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

    def getData(self, url, start, end):
        """The spider get the url, find the "Custom Range" button and insert 
        the starting and ending download period. Then moves back to the graph,
        enable the Options button, select the "Export data" entry of the 
        dropdown menu and finally downloads data.
        If any connection or congestion problem occurs, the process is 
        automatically repeated.
        
        Parameters
        ----------
        url : str
            Terna website url
        start : str
            starting download period
        end : str
            ending download period
        """
        self.driver.get(url)
        self.driver.switch_to.frame(
            self.driver.find_element_by_id("iframeEnergyBal")
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
                    "-child(34) > transform:nth-child(1) > div:nth-child(1) "\
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
                
                # Graph
                graph = self.driver.find_element_by_css_selector(
                    "#pvExplorationHost > div > div > exploration > div "\
                    "> explore-canvas-modern > div > div.canvasFlexBox > div "\
                    "> div.displayArea.disableAnimations.fitToPage "\
                    "> div.visualContainerHost > visual-container-repeat "\
                    "> visual-container-modern:nth-child(23) > transform"
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
                    "/html/body/div[9]/drop-down-list/ng-transclude/"\
                    "ng-repeat[1]/drop-down-list-item/ng-transclude/ng-switch/"\
                    "div"
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
                
                self.getData(url, start, end)
                
                break
            
            except ElementNotInteractableException:
                self.log.error('[TERNA] Interactable error. Try again..')
                
                try:
                    bot('ERROR', 'TERNA', 'Interactable error. Try again..')
                except:
                    pass
                
                self.getData(url, start, end)
                
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
            self.setFname(start)

    def setFname(self, date):
        """When the .xlsx data is correctly downloaded its name is changed into
        EnergyBalDDMMYYY
        
        Parameters
        ----------
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
                    target = Path(f"{DOWNLOAD}/{TERNA['name']}{date}.xlsx")
                    self.log.info(f"[TERNA] {TERNA['name']}{date}.xlsx"\
                        " downloaded."
                    )
                    sleep(2)
                    files.replace(target)
                    
                    QUEUE.put(f"{TERNA['name']}{date}.xlsx")
                    
                    return None
        sleep(1)
