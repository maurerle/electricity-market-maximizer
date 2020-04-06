from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django import forms
from .models import Profile
from influxdb import InfluxDBClient

client = InfluxDBClient(
    'localhost',
    8086,
    'root',
    'root',
    'PublicBids'
)
res = client.query('show tag values with key = op').raw
CHOICE = []
for i in res['series']:
    for j in i['values']:
        obj = (j[1].upper(),j[1].upper())
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
    