# Smart Grids (ML)
A supporting system for strategic bidding in the Italian Electricity Markets through data analytics.

## Description
In the Italian power systems, currently there are 3 most important spot markets, i.e. dayahead market (MGP), Intra-day market (MI) and ancillary service market - MSD. 
This project will be focused on implementing a web-based platform which supports the bidding activity of generation companies in one or more markets resorting to the public available data. 
More specifically, the following tasks are expected from the group:
1. Designing web Crawlers to automatically collect data from public available sources (e.g. Terna, GME, etc.);
2. Selecting suitable database and implement data storage structures and management strategies;
3. Extracting useful information from task 1 to store them in the database designed in step 2;
4. Designing machine learning algorithms to understand the bidding strategies of each market participants in one or more markets;
5. Implement a web-based supporting platform;

## Project Simplified Structure
```
project
|   README.md
|   requirements.txt
├── spiders
|   ├── main_spider.py
|   └── src
|       ├── common
|       |   ├── addcrontab.sh
|       |   ├── config.py
|       |   └── logging.conf
|       ├── database
|       |   ├── processor.py
|       |   ├── csvParse.py
|       |   └── xmlprocessors.py
|       └── spiders
|           ├── gme.py
|           └── terna.py
└── webapp 
|   ├── db.sqlite3
|   ├── manage.py
|   ├── contacts
|   │   └── 
|   ├── dashboard
|   │   ├── backend
|   |   |   ├── arima.py
|   |   |   ├── genetic.py
|   |   |   ├── influxClient.py
|   |   |   ├── intersection.py
|   |   |   └── optimize.py
|   │   └── templates
|   │       └── dashboard
|   |           ├── gme.html
|   |           ├── mi.html
|   |           ├── msd.html
|   |           ├── optimzie.html
|   |           ├── optimized.html
|   |           └── wait.html
|   ├── index
|   │   |   ...
|   │   └── templates
|   │       └── index
|   |           └── index.html
|   ├── register
|   |   |    ...
|   │   └── templates
|   │       └── register
|   |           ├── register.html
|   |           └── login.html
|   ├── smartgridsapp
|   |   |   ...
|   |   ├── settings.py
|   |   └── urls.py
|   └── static
|       └── ...
└── analysis         
    ├── data
    │   └── ...
    ├── fig
    │   └── ...
    ├── models
    │   └── ...
    ├── notebook
    │   ├── arima_err.ipynb
    │   ├── arima.ipynb
    │   ├── arima_training.py
    │   ├── bilateral.ipynb
    │   ├── comparison.ipynb
    │   ├── genetic.ipynb
    │   ├── lstm_corr.ipynb
    │   ├── lstm.ipynb
    │   ├── svr.py
    │   └── svr_training.py
    └── src
        ├── arima.py
        ├── genetic.py
        ├── intersection.py
        └── lstm.py
```

## Technical Details

## Setup
Open a terminal in the project folder and run
```
$ make install
```
## Install geckodrivers for Selenium
Download the latest geckodrivers from ```https://github.com/mozilla/geckodriver/releases``` and copy the file in ```/usr/local/bin```.

### Logs
The log messages are displayed in the console thanks to the logging module. Five levels are used (the WARNING, ERROR and CRITICAL level are also saved in the ``` src/logging.conf ``` file.
#### Usage
At the top of each file use:
```
import logging
import logging.config

logging.config.fileConfig('src/logging.conf')
logger = logging.getLogger(__name__)
```
To display a new log message use:
```
logger.debug('debug message')
logger.info('info message')
logger.warning('warn message')
logger.error('error message')
logger.critical('critical message')
```

## Access to the database
* Using ```src/database/client.py```, follow the instructions on the command-line interface.

* ```src/database/clientarg.py``` accepts command-line arguments and options. Run ```python clientarg.py -h``` to find out more details on its usage.

 ## First Database Creation
 Move to ```~/smartgrids/``` folder, open a terminal and run
 ```
$ make createDatabase
```

## 
(c) 2019, Gian Pio Domiziani, Luca Gioacchini, Francesco Guaiana, Bruno Valente
