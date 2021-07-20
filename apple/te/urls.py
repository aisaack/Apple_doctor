from django.urls import path
from django.conf.urls import include, url
# from rest_framework.urlpatterns import format_suffix_patterns

from . import views

app_name = 'te'

urlpatterns = [
   path('', views.index, name = "index"),
   path('disease/', views.disease, name='disease'),
   path('disease/<slug:slug>/', views.detail, name='detail'),
   path('pesticide/', views.pesticide, name='pesticide'),
   path('api/', include('api.urls')),
   path('pesticide/detail/', views.pesticide_detail, name='pesticide_detail'),
   path('pesticide/list/', views.pest_list, name='pest_list')
] 


# urlpatterns = format_suffix_patterns(urlpatterns)

