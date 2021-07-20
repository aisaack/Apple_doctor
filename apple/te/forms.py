from django import forms

class ResultViewForm(forms.Form):
    image = forms.ImageField()
