FROM python:3.7-slim

WORKDIR /smartgrids/webapp

COPY requirements.txt /smartgrids/webapp
RUN pip3 install --upgrade pip -r requirements.txt

COPY . $WORKDIR

EXPOSE 8000

CMD ./wait-for-it.sh -t 30 172.28.5.1:8086 -- python3 manage.py runserver 172.28.5.2:8000
