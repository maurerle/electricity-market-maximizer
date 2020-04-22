from django import forms

class Production(forms.Form):
    prod_limit = forms.IntegerField(widget=forms.TextInput
        (attrs={
            'placeholder':'[MWh]', 
            'class':'input100'
        }))   