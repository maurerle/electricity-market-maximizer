from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('runModel', views.runModel, name='run'),
    path('dataSource', views.about, name='about')
]