# Smart Grids (ML)
A supporting system for strategic bidding in the Italian Electricity Markets through data analytics.

## Description
In the Italian power systems, currently there are 3 most important spot markets, i.e. dayahead market (MGP), Intra-day market (MI) and ancillary service market - MSD. 
This project will be focused on implimenting a web-based platform which support the bidding activity of generation companies in one or more markets resorting to the public available data. 
More specifically, the following tasks are expected from the group:
1. Design web Crawlers to automatically collecting data from public available sources (e.g. Terna, GME, etc.);
2. Selecting suitable database and implement data storage structures and management strategies;
3. Extracting useful information from task 1 to store them in the database designed in step 2;
4. Design machine learning algorithms to understand the bidding strategies of each market participants in one or more markets;
5. Impliment a web-based supporting plantform;

## Technical Details

## Setup
Open a terminal in the project folder and run
```
# sudo pip3 install -r requirements.txt
```

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
