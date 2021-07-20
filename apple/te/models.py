from typing import Reversible
from django.db import models
import os
from django.db.models.fields import reverse_related
from django.utils import timezone
from uuid import uuid4
from django import forms
from django.utils.text import slugify
from django.urls import reverse
import datetime


# Create your models here.
# 저장할 이미지 종류 정할 것.
def date_upload_to(instance, filename):
    # upload_to="%Y/%m/%d" 처럼 날짜로 세분
    ymd_path = timezone.now().strftime('%Y/%m/%d') 
    # 길이 32 인 uuid 값
    uuid_name = uuid4().hex
    # 확장자 추출
    extension = os.path.splitext(filename)[-1].lower()
    # 결합 후 return
    return 'upload/'.join([ymd_path, uuid_name + extension, ])


class upload_image(models.Model):
    name = models.CharField(max_length=150, default="default_name")
    photo = models.ImageField(upload_to=date_upload_to)
    created = models.DateTimeField(auto_now_add = True)
    def __str__(self):
        return self.name

# 병해 카테고리 페이지 (admin에 category, disease 생성 /url매핑 이름값 문자 위해 slug)
class Category(models.Model):
    name = models.CharField(max_length=250, unique=True, default=None)
    slug = models.SlugField(max_length=250, unique=True, default=None)
    description = models.TextField(blank=True)
    image = models.ImageField(upload_to='category', blank=True)

    class Meta:
        verbose_name = 'category'
        verbose_name_plural = 'categories'

    def __str__(self):
        return '{}'.format(self.name)

class Disease(models.Model):
    name = models.CharField(max_length=250, unique=True, default=None)
    slug = models.SlugField(max_length=250, unique=True, default=None)
    description = models.TextField(blank=True)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    symptom = models.TextField(blank=True)
    pest = models.TextField(blank=True)
    image = models.ImageField(upload_to='disease', blank=True)

    class Meta:
        verbose_name = 'disease'
        verbose_name_plural = 'diseases'

    def __str__(self):
        return '{}'.format(self.name)


class Pest(models.Model):
#0   pestiKorName  1   regCpntQnty  2   pestiBrandName  3   compName  4   toxicName 
# 5 fishToxicGubun  6   useName  7   indictSymbl  8   cropName   9   diseaseWeedName 
#10  pestiUse  11  dilutUnit   12  useSuittime  13  useNum 
    pestiKorName = models.TextField(verbose_name='품목명', blank=True)
    regCpntQnty = models.TextField(verbose_name='주성분함량', blank=True)
    pestiBrandName = models.TextField(verbose_name='제품명', blank=True)
    compName = models.TextField(verbose_name='제조사', blank=True)
    useName = models.TextField(verbose_name='용도', blank=True)
    indictSymbl = models.TextField(verbose_name='작용기작', blank=True)
    cropName = models.CharField(verbose_name='작물', max_length=250,)
    diseaseWeedName = models.CharField(verbose_name='적용병해충', max_length=250, default=None)
    pestiUse = models.TextField(verbose_name='사용적기 및 방법', blank=True)
    dilutUnit = models.CharField(verbose_name='희석배수(10a당 사용량)' , max_length=250)
    useSuittime = models.CharField(verbose_name='안전사용시기', max_length=250,)
    useNum = models.CharField(verbose_name='안전사용횟수', max_length=250)

    def __str__(self):
        return self.diseaseWeedName

