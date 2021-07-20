from typing import List
from django.shortcuts import get_object_or_404, render, redirect
from django.template import RequestContext
from django.http import HttpResponse
from django.template import context
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
# from .models import upload_imageForm
from .forms import ResultViewForm
from api.serializers import upload_imageSerializers
from te.models import upload_image

import colorsys
import cv2
import os
import sys
from skimage.measure import find_contours
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
from matplotlib import patches, lines
from matplotlib.patches import Polygon
import json
from tensorflow import Graph, Session
# Create your views here.
from django.core.files.storage import FileSystemStorage
import skimage.io
import tensorflow as tf
from te.model_d.mrcnn import model as modellib
import te.model_d.apple as apple
from te.model_d.mrcnn import visualize
from .crawling import agchmParsing
from .models import  Disease, Pest
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator, EmptyPage,InvalidPage
from django.views.generic import TemplateView, ListView
from .models import Pest
import math
# Create your views here.

import tensorflow as tf
from te.model_d.mrcnn import model as modellib,utils
import te.model_d.apple as apple
import requests
from django.core.paginator import Paginator
import logging
logger = logging.getLogger(__name__)
import requests
webAPI = "http://192.168.1.37:81/api/web/predict"

def my_view(request):
    logger.debug(f'{self.request.user}')


@csrf_exempt
def index(request):
    if request.method == "POST":
        image = request.FILES.get('photo')
        data = requests.post(webAPI, files={'photo':image}).json()

        # print(json)
        return render(request, 'devfo/index.html', data)
    else:
        return render(request, 'devfo/index.html')   # verti/index.html

    
def pesticide(request):
    return render(request, 'devfo/index.html')   # verti/pesticide_info.html
 
#조


def pesticide_detail(request):
    """
    pesticide 농약 내용 출력
    """
    pest_list = Pest.objects.order_by('diseaseWeedName')
    page = request.GET.get('page',1) #페이지
    paginator = Paginator(pest_list,50)
    page_obj = paginator.get_page(page)

    context = {'pest_list':page_obj}
    return render(request, 'devfo/index.html', context)   # verti/pesticide_detail.html

# 조
def disease(request):
    """
    disease 목록 출력
    """
    disease_list = Disease.objects.order_by('name')
    context = {'disease_list':disease_list}
    return render(request, 'devfo/index.html', context)   # verti/disease_list.html

   
def detail(request, slug):
    """
    disease 내용 출력
    """
    disease = get_object_or_404(Disease, slug=slug)
    print(disease)
    context = {'disease':disease}
    return render(request, 'devfo/index.html', context)     #  verti/detail.html

def pest_list(request):
    search_key = request.GET.get('search_key')
    print(search_key)
    pest_list = Pest.objects.all()
    if search_key:
        pest_list = pest_list.filter(diseaseWeedName__icontains=search_key)
    if len(pest_list) == 0 :
        return render(request, 'devfo/index.html', {'pest_list':{"non":"죄송합니다. 해당하는 정보가 없습니다."}})
    else:
        page = request.GET.get('page')
        paginator = Paginator(pest_list, 10)
        boards = paginator.get_page(page)
        return render(request, 'devfo/index.html',{"pest_list": boards,"search_key":search_key})




   