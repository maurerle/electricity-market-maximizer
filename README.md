# GME Profit Maximizer
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

## Setup
If you don't have docker-compose installed on your machine type  
```
$ sudo apt-get install docker-compose -y
$ sudo apt-get update
$ sudo apt-get upgrade -y
```  
Once installed open a terminal in the project root folder and run  
```
$ cd project/
```  
Now you have to build the full image of the application, so type  
```
$ sudo docker-compose up --build --no-start
```  
The services are now configurated type  
```  
$ sudo docker-compose up 
```  
to start the application in verbose mode, or  
```  
$ sudo docker-compose up -d
```  
to execute the processes in background.  
The web application should be reachable from a browser at ```http://172.28.5.2:8000/```.  

In case of network bridge error:
``` 
ERROR: Service 'webapp' failed to build: failed to create endpoint stoic_wright on network bridge: network e6c...493 does not exist
``` 
Try to upgrade Docker. If the error persists try
``` 
$ sudo docker system prune
``` 
but pay attention. This instruction will remove all the previously built docker images.

**NOTE:** To save space, the local database natively stores data from the 01/03/2020 up to 14/05/2020. To add all the available data from the 01/02/2017 see the second paragraph of Web Crawlers in the Examples sections.  
Since the Italian regulation publish GME data up to one week before the current days, the platform current day is the one of a month before the current one for dimostrative purpose.

## Examples
The web crawlers and the prediction module are scheduled to run at 00:30 of each day. If you want to check their stand-alone functioning open a terminal in the project root folder and run  
```
$ cd project/example/
```   
**NOTE:** At the end of each example the original Dockerfile should be automatically restored. A confirm of this should be obtained by the ```Original Dockerfile restored``` message. However, to be sure, you can run from the ```example/``` folder  
```./restore_original.sh```.

### Web Crawlers
Two functions are developed in the spiders module.  
1. The first one download and store the daily market data. To check it, from the ```example/``` folder type

```
$ ./spider_daily.sh
```

2. The second function allows to create the database from scratch by downloading all available data from the 01/02/2017. To check it, from the ```example/``` folder type  

```
$ ./create_database.sh
```

### Prediction
To run the prediction module in stand-alone mode, from the ```example/``` folder type  

```
$ ./predict.sh
```

## 
(c) 2020, Gian Pio Domiziani, Luca Gioacchini, Francesco Guaiana, Bruno Valente

