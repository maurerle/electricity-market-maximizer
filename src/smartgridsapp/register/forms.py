from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django import forms
from .models import Profile
from influxdb import InfluxDBClient
from datetime import datetime
from dateutil import relativedelta

now = datetime.now()
lastMonth = now - relativedelta.relativedelta(months=6)
lastMonth = int(datetime.timestamp(lastMonth)*1e9)
CHOICE = []
client = InfluxDBClient(
    'localhost',
    8086,
    'root',
    'root',
    'PublicBids'
)

for market in ['MGP', 'MI', 'MSD']:
    res = client.query(f"SELECT * FROM demand{market} WHERE time >= {lastMonth}").raw
    for val in res['series'][0]['values']:
        obj = (val[3], val[3])
        if obj not in CHOICE:
            CHOICE.append(obj)

    res = client.query(f"SELECT * FROM supply{market} WHERE time >= {lastMonth}").raw
    for val in res['series'][0]['values']:
        obj = (val[3], val[3])
        if obj not in CHOICE:
            CHOICE.append(obj)

class CreateUserForm(UserCreationForm):
    username = forms.CharField(widget=forms.TextInput
        (attrs={
            'placeholder':'Enter your first name', 
            'class':'input100'
        }))
    password1 = forms.CharField(widget=forms.PasswordInput
        (attrs={
            'placeholder':'Enter your password', 
            'class':'input100'
        }))
    password2 = forms.CharField(widget=forms.PasswordInput
        (attrs={
            'placeholder':'Retype your password', 
            'class':'input100'
        }))
    
    operator = forms.CharField(widget=forms.Select
        (attrs={
            'placeholder':'Choose your company', 
            'class':'input100'
        }, choices=CHOICE))
