getDaily:
	@python3 -c 'from src.mainSpider import getDay; getDay()'

createDatabase:
	@python3 -c 'from src.mainSpider import getHistory; getHistory()'

install:
	@sudo pip3 install -r requirements.txt
	@src/common/addcrontab.sh
