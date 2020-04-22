from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django import forms
from .models import Profile
from influxdb import InfluxDBClient
CHOICE = []

class CreateUserForm(UserCreationForm):
    client = InfluxDBClient(
        '172.28.5.1',
        8086,
        'root',
        'root',
        'PublicBids'
    )
    
    for market in ['MGP', 'MI', 'MSD']:
        res = client.query(f"SELECT * FROM demand{market}").raw
        for val in res['series'][0]['values']:
            obj = (val[3], val[3])
            if obj not in CHOICE:
                CHOICE.append(obj)

        res = client.query(f"SELECT * FROM supply{market}").raw
        for val in res['series'][0]['values']:
            obj = (val[3], val[3])
            if obj not in CHOICE:
                CHOICE.append(obj)

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
