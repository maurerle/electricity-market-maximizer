import os
import requests as req
from datetime import date
from lxml import html
import src.common.config as conf
import logging
import logging.config

# logging.config.fileConfig('src/logging.conf')
# logger = logging.getLogger(__name__)

today = date.today().strftime('%d.%m.%Y')
path = "C:\\Users\franc\Desktop\ICT\Primo Semestre II\Interdisciplinary projects\repo\smartgrids\downloads"
payload1 = {
    'name': '',
    'defaultValue': 'false',
    'viewType': 'TABLE',
    'areaType': 'CTA',
    'atch': 'false',
    'dateTime.dateTime': today + ' 00:00|CET|DAY',
    'biddingZone': 'CTY|10YIT-GRTN-----B!CTA|10YIT-GRTN-----B',  # CTY%7C10YIT-GRTN-----B!CTA%7C10YIT-GRTN-----B
    'dateTime.timezone': 'CET_CEST',
    'dateTime.timezone_input': 'CET (UTC 1) / CEST (UTC 2)',  # CET+(UTC%2B1)+%2F+CEST+(UTC%2B2)
    'dataItem': 'ACTUAL_TOTAL_LOAD',
    'timeRange': 'DEFAULT',
    'exportType': 'CSV'

}
URL = f"https://transparency.entsoe.eu/load-domain/r2/totalLoadR2/export?name={payload1['name']}&defaultValue={payload1['defaultValue']}&view" \
      f"Type={payload1['viewType']}&areaType={payload1['areaType']}&atch={payload1['atch']}&dateTime.dateTime={today}+00%3A00%7CCET%7CDAY&biddingZone." \
      f"values=CTY%7C10YIT-GRTN-----B!CTA%7C10YIT-GRTN-----B&dateTime.timezone={payload1['dateTime.timezone']}&dateTime.timezone_input=CET+(UTC%2B1)+%2F+CEST+(UTC%2B2)&dataItem=" \
      f"{payload1['dataItem']}&timeRange={payload1['timeRange']}&exportType={payload1['exportType']}"


class EntsoeSpider():
    def __init__(self):
        # Class init

        self.login()

    def login(self):
        today = date.today().strftime('%d.%m.%Y')
        session_requests = req.session()
        result = session_requests.post(conf.LOGIN_URL)
        # f = open("dashboard.html", "w")
        # f.write(result.text)
        # f.close()
        # webbrowser.open("dashboard.html")

        print("first get: " + str(result.ok))
        print("SC :" + str(result.status_code))
        self.getData(session_requests)

    def getData(self, session_requests):
        result = session_requests.get(URL)
        with open("CTA.csv", 'wb') as f:  # csv or #xml ?  #non riesco a salvarlo in "download"
            f.write(result.content)

        tree = html.fromstring(result.content)  # ?
        print("second get: " + str(result.ok))
        print("SC :" + str(result.status_code))

        # Retrieve HTTP meta-data
        print(result.status_code)
        print(result.headers['content-type'])
        print(result.encoding)


def main():
    spider = EntsoeSpider()


if __name__ == '__main__':
    main()
