from django.contrib import admin
from django.urls import path
from api import views
from rest_framework.urlpatterns import format_suffix_patterns


urlpatterns = [
    path('web/predict', views.webAPI.as_view(), name = 'web_post'),
    path('mobile/predict', views.mobileAPI.as_view(), name = 'mobile_post'),
    path('get/<int:pk>/', views.getAPI.as_view(), name = 'get')
]

urlpatterns = format_suffix_patterns(urlpatterns)