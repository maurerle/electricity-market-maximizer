getDaily:
	@python3 -c 'from src.mainSpider import getDay; getDay()'

createDatabase:
	@python3 -c 'from src.mainSpider import getHistory; getHistory()'

runServer:
	sudo fuser -k 8000/tcp
	@python3 src/webapp/manage.py runserver

install:
	@sudo pip3 install -r requirements.txt
	@src/common/addcrontab.sh
	make runServer