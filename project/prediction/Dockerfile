FROM python:3.7-slim

WORKDIR smartgridsapp/prediction

# Install os requirements
RUN apt-get update && apt-get install -y cron

# Install python requirements
COPY requirements.txt $WORKDIR
RUN pip3 install -r requirements.txt

COPY . $WORKDIR

# Add crontab for root and schedule the spider
RUN crontab
RUN crontab -l > crontab
RUN echo 'PATH=/usr/local/bin/:/usr/bin:/usr/sbin' >> crontab
RUN touch /smartgridsapp/prediction/crontab.log
RUN echo "30 22 * * * /smartgridsapp/prediction/predict.sh >> /smartgridsapp/prediction/crontab.log 2>&1" >> crontab
RUN crontab crontab

CMD ./wait-for-it.sh -t 30 172.28.5.1:8086 -- cron -f
